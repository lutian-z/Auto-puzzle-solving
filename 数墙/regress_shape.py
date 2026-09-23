# -*- coding: utf-8 -*-
"""regress_shape.py — 数墙"任意形状直通"离线回归(常驻脚本, 不依赖屏幕).

背景: 2026-09-23 移除识别器对 solver.shape_params('矩形-四角裁-中间裁')的
硬校验, 改为存在矩阵直通 + 四连通校验。本脚本守住四条底线:
  1) 老的可参数化形状识别行为与改造前逐位一致(对拍备份代码);
  2) 自由异形(M/螺旋等)在旧代码下确实被模板拒绝、新代码识别+求解通过;
  3) 存在格对角不相接(四连通不成立)的形状被明确拒绝;
  4) 真实截图素材端到端识别+求解+独立校验。

用法:
    python 数墙/regress_shape.py        # 全量回归, 退出码 0=全部通过
依赖: opencv numpy Pillow ortools(可选, 缺失时自动走自研求解回退路径).
仅离线测试用, 不参与主流程; 不 import main/automation(避免屏幕依赖).
"""
import ast
import os
import random
import sys
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 沙箱 cv2 5.x 垫片: HoughLinesP 返回 (N,4) 而非 4.x 的 (N,1,4).
# 用户机 cv2 4.x 下本垫片是恒等操作。
_orig_hough = cv2.HoughLinesP


def _hough(*a, **k):
    r = _orig_hough(*a, **k)
    if r is not None and getattr(r, "ndim", 3) == 2:
        r = r[:, None, :]
    return r


cv2.HoughLinesP = _hough

import recognizer                      # noqa: E402
import sat_engine                      # noqa: E402
import solver                          # noqa: E402

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FONT_CANDS = [
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
BAK = os.path.join(BASE, "recognizer.py.bak3_20260923")


def load_cfg():
    """从 main.py 源码字面提取 CFG(避免 import main 拉起屏幕依赖)."""
    with open(os.path.join(BASE, "main.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "CFG"):
            return eval(ast.unparse(node.value),
                        {"os": os, "BASE_DIR": BASE})
    raise RuntimeError("main.py 中未找到 CFG 字典")


def load_old_recognizer():
    """加载改造前备份, 用于逐位一致对拍; 缺失时返回 None."""
    if not os.path.exists(BAK):
        return None
    import types
    mod = types.ModuleType("recognizer_old")
    mod.__file__ = BAK
    sys.modules["recognizer_old"] = mod
    with open(BAK, encoding="utf-8") as f:
        src = f.read()
    exec(compile(src, BAK, "exec"), mod.__dict__)   # .bak 后缀走不了 import 机制
    return mod


# 形状构造

def shape_classic():
    """矩形+四角裁+内部空洞 — 旧'通用形式'覆盖的形状(基线行为必须不变).

    所有裁口两个维度均 >=2 格, 且不产生"单格宽凸出"的外露短边: 这类短
    线段短于线-开运算核, 会残留在相邻格内被识别成假字形(既有怪癖, 与形
    状直通改造无关, 真实游戏素材不产生该结构). 内部空洞四周线都被两侧存
    在格共享, 故安全.
    """
    R, C = 8, 10
    s = [[True] * C for _ in range(R)]
    for r in range(2):                    # 四角各 2x2
        for c in range(2):
            s[r][c] = False               # 左上
            s[r][C - 2 + c] = False       # 右上
            s[R - 2 + r][c] = False       # 左下
            s[R - 2 + r][C - 2 + c] = False   # 右下
    for r in (3, 4):                      # 内部 2x2 空洞(参数化 middle)
        for c in (3, 4):
            s[r][c] = False
    return s


def shape_free():
    """M 形: 双联竖块+中段走廊(与用户截图同构), 不满足旧参数化形式."""
    R, C = 12, 11
    s = [[False] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if r <= 3 or r >= 8:
                s[r][c] = not (3 <= c <= 5)
            elif 4 <= r <= 6:
                s[r][c] = 2 <= c <= 8
            else:                      # r == 7
                s[r][c] = True
    return s


def shape_spiral():
    """S/螺旋形 — 另一种自由形状."""
    R, C = 9, 9
    s = [[False] * C for _ in range(R)]
    for r in range(0, 4):
        for c in range(0, 5):
            s[r][c] = True
    for r in range(2, 7):
        for c in range(2, 7):
            s[r][c] = True
    for r in range(5, 9):
        for c in range(4, 9):
            s[r][c] = True
    return s


def shape_diag():
    """两个 3x3 块仅对角点相接 — 四连通不成立, 必须被拒绝."""
    s = [[False] * 6 for _ in range(6)]
    for r in range(3):
        for c in range(3):
            s[r][c] = True
    for r in range(3, 6):
        for c in range(3, 6):
            s[r][c] = True
    return s


# 合成图渲染与必有解题面构造

def render(shape, nums, cell=40, line=2, pad=30):
    """白底黑线渲染题目图(BGR), 不存在的格完全留白(与背景连通)."""
    R, C = len(shape), len(shape[0])
    w, h = C * cell + line, R * cell + line
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    d = ImageDraw.Draw(img)
    fp = next((p for p in FONT_CANDS if os.path.exists(p)), None)
    if fp is None:
        raise RuntimeError("沙箱缺 PIL 字体, 无法合成题目图")
    font = ImageFont.truetype(fp, int(cell * 0.62))
    for r in range(R):
        for c in range(C):
            if not shape[r][c]:
                continue
            x0, y0 = c * cell + pad, r * cell + pad
            d.rectangle([x0, y0, x0 + cell + line - 1, y0 + cell + line - 1],
                        outline=0, width=line)
    for (r, c), v in nums.items():
        x0, y0 = c * cell + pad, r * cell + pad
        d.text((x0 + (cell + line) // 2, y0 + (cell + line) // 2),
               str(v), fill=0, font=font, anchor="mm")
    gray = np.array(img)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_solvable(shape, rng, tries=5000):
    """随机造该形状下必有解的题: 随机黑海 -> 拆2x2 -> 取最大连通块.

    规则映射: 黑=海(取最大连通分量保证规则2), 白色连通块面积=数字
    (规则1/4), 逐个拆掉 2x2 全黑(规则3, 挖黑不会新增2x2)。构造完用
    solver.verify_shape 自检, 不合格整体重试; 返回的必是一道合法题面
    与它的一个真解。
    """
    R, C = len(shape), len(shape[0])
    cells = sorted((r, c) for r in range(R) for c in range(C) if shape[r][c])
    cset = set(cells)

    def nbrs(p):
        return [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                if (p[0] + dr, p[1] + dc) in cset]

    def largest_cc(subset):
        seen, best = set(), set()
        for s in cells:
            if s in subset and s not in seen:
                comp, stack = {s}, [s]
                seen.add(s)
                while stack:
                    p = stack.pop()
                    for q in nbrs(p):
                        if q in subset and q not in seen:
                            seen.add(q)
                            comp.add(q)
                            stack.append(q)
                if len(comp) > len(best):
                    best = comp
        return best

    for _try in range(tries):
        black = {c for c in cells if rng.random() < rng.uniform(0.52, 0.68)}
        if not black:
            continue
        for _fix in range(4 * len(cells)):
            bad = [(r, c) for r in range(R - 1) for c in range(C - 1)
                   if all((r + dr, c + dc) in black
                          for dr in (0, 1) for dc in (0, 1))]
            if not bad:
                break
            r, c = bad[rng.randrange(len(bad))]
            victim = [(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)]
            black.discard(victim[rng.randrange(4)])
        else:
            continue
        black = largest_cc(black)          # 海必须连通
        if len(black) < 0.2 * len(cells):
            continue
        seen, nums, ok = set(), {}, True
        for start in cells:
            if start in black or start in seen:
                continue
            comp, stack = [start], [start]
            seen.add(start)
            while stack:
                p = stack.pop()
                for q in nbrs(p):
                    if q not in black and q not in seen:
                        seen.add(q)
                        comp.append(q)
                        stack.append(q)
            if len(comp) > 16:
                ok = False
                break
            nums[min(comp)] = len(comp)
        if not ok or not nums:
            continue
        truth = [[bool(shape[r][c]) and (r, c) in black
                  for c in range(C)] for r in range(R)]
        good, _errs = solver.verify_shape(shape, nums, truth)
        if good:
            return nums, truth
    raise AssertionError("造题失败(多次重试仍不满足数墙规则)")


# 断言辅助

def _fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def check_roundtrip(name, shape, nums, cfg, old_mod=None,
                    expect_params=None):
    """渲染→识别: 存在矩阵必须逐位一致; 数字与 params 按策略断言.

    old_mod 给出时(基线用例): shape/nums/params 与改造前逐位一致即可,
    不要求等于题面给定值 — 数字识别的既有怪癖新旧同有, 语义不变才算过。
    否则(自由形状用例): 断言 nums 精确一致。
    """
    img = render(shape, nums)
    try:
        puz = recognizer.recognize(img, cfg)
    except recognizer.RecognizeError as e:
        return _fail(f"{name}: 识别失败: {e}")
    if puz.shape != shape:
        diff = [(r, c) for r in range(len(shape)) for c in range(len(shape[0]))
                if bool(puz.shape[r][c]) != bool(shape[r][c])]
        return _fail(f"{name}: 存在矩阵不一致 {diff[:10]}")
    if old_mod is not None:
        try:
            opuz = old_mod.recognize(img, cfg)
        except Exception as e:
            return _fail(f"{name}: 可参数化形状旧版竟识别失败: {e}")
        if (puz.shape, puz.nums, puz.params) != \
                (opuz.shape, opuz.nums, opuz.params):
            return _fail(f"{name}: 与改造前行为不一致!\n"
                         f"       新nums={sorted(puz.nums.items())}\n"
                         f"       旧nums={sorted(opuz.nums.items())}\n"
                         f"       新params={puz.params}\n"
                         f"       旧params={opuz.params}")
        tag = " [新旧逐位一致]"
    else:
        tag = ""
        if puz.nums != nums:
            return _fail(f"{name}: 数字不一致 期望{sorted(nums.items())} "
                         f"实得{sorted(puz.nums.items())}")
    if expect_params is True and puz.params is None:
        return _fail(f"{name}: 可参数化形状 params 变为 None(行为回归)")
    print(f"  [ok] {name}: {puz.describe()}{tag}")
    return True


def check_old_rejects_free(cfg, old_mod):
    """反向对照: 自由异形在旧代码下必须被模板门槛拒绝."""
    if old_mod is None:
        print("  [skip] 备份 recognizer.py.bak3_20260923 不在, 跳过反向对照")
        return True
    shape = shape_free()
    nums, _ = build_solvable(shape, random.Random(7))
    img = render(shape, nums)
    try:
        old_mod.recognize(img, cfg)
    except Exception as e:      # 旧模块的 RecognizeError 是另一个类, 按消息判
        if "通用形式" in str(e):
            print(f"  [ok] 旧代码按模板门槛拒绝自由形状: {e}")
            return True
        return _fail(f"旧代码拒绝原因异常: {e}")
    return _fail("旧代码竟然接受了自由形状 — 对照失效")


def check_solve(name, shape, nums, cfg, truth=None, budget=30.0):
    t0 = time.time()
    status, grid, _info = sat_engine.solve_auto(shape, nums,
                                                time_budget=budget)
    dt = time.time() - t0
    if status != "ok":
        return _fail(f"{name}: 求解失败 status={status} ({dt:.1f}s)")
    ok, errs = solver.verify_shape(shape, nums, grid)
    if not ok:
        return _fail(f"{name}: 解未通过独立校验: {errs[:3]}")
    if truth is not None:
        ok_t, errs_t = solver.verify_shape(shape, nums, truth)
        if not ok_t:
            return _fail(f"{name}: 构造的真解未过校验(造题器坏了): {errs_t[:3]}")
    print(f"  [ok] {name}: 求解+独立校验通过 ({dt:.2f}s)")
    return True


def check_pipeline(name, shape, rng, cfg, old_mod=None):
    """造必有解题 → 渲染识别直通 → 求解 → 双重独立校验."""
    nums, truth = build_solvable(shape, rng)
    r1 = check_roundtrip(f"{name}-识别", shape, nums, cfg, old_mod=old_mod,
                         expect_params=False)
    if not r1:
        return False
    img = render(shape, nums)
    puz = recognizer.recognize(img, cfg)
    return check_solve(f"{name}-求解", puz.shape, puz.nums, cfg, truth=truth)


def main():
    cfg = load_cfg()
    old = load_old_recognizer()
    rng = random.Random(20260923)
    results = []

    print("[case 1] 老的可参数化形状: 新旧代码识别结果逐位一致(基线不破)")
    n1 = {(1, 3): 3, (2, 0): 7, (3, 9): 11, (7, 2): 4, (5, 5): 2, (6, 3): 6}
    results.append(check_roundtrip("classic", shape_classic(), n1, cfg,
                                   old_mod=old, expect_params=True))

    print("[case 2] 反向对照: 自由形状在旧代码下被模板拒绝(修复确有必要)")
    results.append(check_old_rejects_free(cfg, old))

    print("[case 3] M 形自由盘(与用户截图同构): 直通识别 + 求解 + 独立校验")
    results.append(check_pipeline("free-M", shape_free(), rng, cfg))

    print("[case 4] 螺旋自由盘: 直通识别 + 求解 + 独立校验")
    results.append(check_pipeline("free-spiral", shape_spiral(), rng, cfg))

    print("[case 5] 对角相接(四连通不成立): 明确拒绝")
    img = render(shape_diag(), {(0, 0): 2, (5, 5): 2})
    try:
        recognizer.recognize(img, cfg)
        results.append(_fail("对角相接形状未被拒绝!"))
    except recognizer.RecognizeError as e:
        results.append(True if "不连通" in str(e)
                       else _fail(f"拒绝原因异常: {e}"))

    print("[case 6] 随机可解自由盘 x2: 造题→识别→求解全链路")
    results.append(check_pipeline(f"rand-A", shape_free(),
                                  random.Random(101), cfg))
    results.append(check_pipeline(f"rand-B", shape_spiral(),
                                  random.Random(202), cfg))

    mat = os.path.join(BASE, "测试素材", "异形M_20260923.png")
    print("[case 7] 真实截图素材(用户提供, 存在则跑端到端)")
    if not os.path.exists(mat):
        print("  [skip] 素材缺失(把截图放到 测试素材/ 可启用)")
        results.append(True)
    else:
        img = cv2.imread(mat)
        if img is None:
            results.append(_fail("素材读取失败"))
        else:
            try:
                puz = recognizer.recognize(img, cfg)
                print(f"  [ok] 真实素材识别: {puz.describe()}")
                results.append(check_solve("真实素材", puz.shape, puz.nums,
                                           cfg, budget=60.0))
            except recognizer.RecognizeError as e:
                results.append(_fail(f"真实素材识别失败: {e}"))

    n_ok = sum(1 for r in results if r)
    print(f"\n== 回归结果: {n_ok}/{len(results)} 通过 ==")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
