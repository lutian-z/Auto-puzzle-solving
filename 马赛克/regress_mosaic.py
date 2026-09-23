# -*- coding: utf-8 -*-
"""regress_mosaic.py — 马赛克"浅网格线放宽重试"离线回归(常驻, 不依赖屏幕).

背景: 2026-09-23 实机 50×50 题在浏览器小数缩放下, 交替网格线被反锯齿
抬到 181~198 灰(格底 204 只差 6 灰阶), 主线色窗口漏掉近半线条 → 报
"竖线间距不均匀(变异系数 0.35)"。修复 = 几何失败时自动放宽线窗重试一次
(_line_mask widen)。本脚本守两条底线:
  1) 普通盘主路径与改造前逐位一致(n/nums/black/centers 全等);
  2) 浅线盘: 旧代码必须报错、新代码必须识别正确且可解, 回读/锚点同步救活;
  3) 无题图(纯噪声)不许被放宽窗"误救"。

用法:
    python 马赛克/regress_mosaic.py      # 退出码 0=全部通过
依赖: opencv numpy Pillow; 求解用例需要 scipy 或纯 Python 回退(慢但可用).
不 import main/automation(避免屏幕依赖), CFG 从 main.py 源码字面提取。
"""
import ast
import logging
import os
import random
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 沙箱 cv2 5.x 垫片(HoughLinesP 返回形状); 用户机 4.x 下为恒等
_orig_hough = cv2.HoughLinesP


def _hough(*a, **k):
    r = _orig_hough(*a, **k)
    if r is not None and getattr(r, "ndim", 3) == 2:
        r = r[:, None, :]
    return r


cv2.HoughLinesP = _hough

import recognizer                       # noqa: E402
import sat_engine                       # noqa: E402
import solver                           # noqa: E402

BAK = os.path.join(BASE, "recognizer.py.bak2_20260923")
FONT_CANDS = [
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
LOG = logging.getLogger("regress")
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.CRITICAL)

V_BG, V_DARK_LINE, V_PALE_LINE, V_INK = 204, 102, 186, 20


def load_cfg():
    with open(os.path.join(BASE, "main.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "CFG"):
            return eval(ast.unparse(node.value), {"os": os, "BASE": BASE})
    raise RuntimeError("main.py 中未找到 CFG")


def load_old_recognizer():
    if not os.path.exists(BAK):
        return None
    import types
    mod = types.ModuleType("recognizer_old")
    mod.__file__ = BAK
    sys.modules["recognizer_old"] = mod
    with open(BAK, encoding="utf-8") as f:
        exec(compile(f.read(), BAK, "exec"), mod.__dict__)
    return mod


# 题面与渲染

def build_truth(n, rng):
    """随机黑格 → 数字=3x3邻域黑数; 要求唯一解(solve_auto 验证), 否则换种子."""
    for _ in range(60):
        black = {(r, c) for r in range(n) for c in range(n)
                 if rng.random() < 0.35}
        nums = {}
        for r in range(n):
            for c in range(n):
                k = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if 0 <= r + dr < n and 0 <= c + dc < n
                        and (r + dr, c + dc) in black)
                if rng.random() < 0.6:
                    nums[(r, c)] = k
        covered = set()
        for (r, c), k in nums.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if 0 <= r + dr < n and 0 <= c + dc < n:
                        covered.add((r + dr, c + dc))
        if len(covered) < n * n:
            continue
        st, grid, _info = sat_engine.solve_auto(n, nums, time_budget=10)
        if st == "ok":
            return nums, black
    raise AssertionError("造题失败: 连续 60 次无法凑出唯一解题面")


def render(n, nums, black, pale=False, cell=28, pad=25, pale_seed=3,
           pale_core=V_PALE_LINE):
    """渲染马赛克题图: 格底 V_BG, 深线 V_DARK_LINE, pale=True 时随机部分线浅化.

    浅线模拟浏览器小数缩放的反锯齿漂白(50×50 实拍: 线芯 181~198, 格底
    204): 浅线 = 1px 芯 pale_core + 左邻 1px pale_core+12。浅化必须按
    真实形态随机分布(p≈0.5) — 若严格隔一条浅一条, 缺失线步长均匀
    (2x格宽), cv 闸不响, 连旧代码都能'成功'把 12x12 认成 6x6 双倍格,
    对照失效。
    """
    pitch = cell + 1
    w = h = n * pitch + 1
    prng = random.Random(pale_seed)
    img = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    img[pad:pad + h, pad:pad + w] = V_BG
    for k in range(n + 1):
        is_pale = bool(pale and prng.random() < 0.5)
        v = pale_core if is_pale else V_DARK_LINE
        img[pad + k * pitch:pad + k * pitch + 1, pad:pad + w] = v
        img[pad:pad + h, pad + k * pitch:pad + k * pitch + 1] = v
        if is_pale:
            if k > 0:
                img[pad + k * pitch - 1:pad + k * pitch, pad:pad + w] = \
                    min(255, v + 12)
                img[pad:pad + h,
                    pad + k * pitch - 1:pad + k * pitch] = min(255, v + 12)
    fp = next((p for p in FONT_CANDS if os.path.exists(p)), None)
    if fp is None:
        raise RuntimeError("无可用字体渲染数字")
    font = ImageFont.truetype(fp, int(cell * 0.62))
    pim = Image.fromarray(img)
    d = ImageDraw.Draw(pim)
    for (r, c) in black:                        # 先涂全部黑格(含无数字的)
        x0 = pad + c * pitch + 1
        y0 = pad + r * pitch + 1
        pim.paste(0, (x0, y0, x0 + cell, y0 + cell))
    for (r, c), k in nums.items():              # 再画字: 黑格反白/灰格深字
        is_b = (r, c) in black
        x0 = pad + c * pitch + 1
        y0 = pad + r * pitch + 1
        d.text((x0 + cell // 2, y0 + cell // 2), str(k),
               fill=250 if is_b else V_INK, font=font, anchor="mm")
    return cv2.cvtColor(np.array(pim), cv2.COLOR_GRAY2BGR)


def _snap(puz):
    return (puz.n, dict(puz.nums), set(puz.black_cells),
            {rc: (round(x, 2), round(y, 2))
             for rc, (x, y) in puz.centers.items()})


def _fail(msg):
    print(f"  [FAIL] {msg}")
    return False


# 用例

def case_normal_identical(cfg, old):
    """普通深线盘: 新旧逐位一致(主路径不许有一丝变化)."""
    rng = random.Random(5)
    n = 9
    nums, black = build_truth(n, rng)
    img = render(n, nums, black, pale=False)
    new = recognizer.recognize(img, cfg, LOG)
    ok = new.n == n and dict(new.nums) == nums
    if not ok:
        return _fail(f"普通盘识别基线错误 n={new.n} "
                     f"nums_ok={dict(new.nums) == nums}")
    if old is None:
        print("  [skip] 备份缺失, 无法对拍")
        return True
    opuz = old.recognize(img, cfg, LOG)
    if _snap(new) != _snap(opuz):
        return _fail("普通盘上改造前后结果不一致! (主路径被改动)")
    print(f"  [ok] 普通 {n}x{n} 盘: 新旧逐位一致 (n/nums/black/centers)")
    return True


def case_pale_rescued(cfg, old):
    """浅线盘: 旧代码必须报错, 新代码识别正确且可解."""
    rng = random.Random(11)
    n = 12
    nums, black = build_truth(n, rng)
    img = render(n, nums, black, pale=True)
    if old is not None:
        old_wrong = False
        try:
            opuz = old.recognize(img, cfg, LOG)
            old_wrong = (opuz.n != n or dict(opuz.nums) != nums)
        except Exception as e:      # 旧模块抛它自己的 RecognitionError 类
            old_wrong = True
            print(f"  [ok] 旧代码在浅线盘上报错(对照成立): {e}")
        if not old_wrong:
            return _fail("浅线盘旧代码竟然全对 — 合成图不够逼真, 对照失效")
    new = recognizer.recognize(img, cfg, LOG)
    if new.n != n or dict(new.nums) != nums or set(new.black_cells) != black:
        return _fail(f"浅线盘新代码识别错误: n={new.n} "
                     f"nums={dict(new.nums) == nums} "
                     f"black={set(new.black_cells) == black}")
    if not any("放宽" in w for w in new.warnings):
        return _fail("浅线盘成功但未见'放宽重试'告警(走的哪条路?)")
    st, grid, _info = sat_engine.solve_auto(new.n, new.nums, time_budget=60)
    if st != "ok":
        return _fail(f"浅线盘求解失败: {st}")
    ok, errs = solver.verify(new.n, new.nums, grid)
    if not ok:
        return _fail(f"浅线盘解未过校验: {errs[:3]}")
    print(f"  [ok] 浅线 {n}x{n} 盘: 新代码识别全对 + 求解校验通过")
    return True


def case_pale_side_entries(cfg):
    """浅线盘上锚点定位与回读(作答阶段入口)也必须被救活."""
    rng = random.Random(13)
    n = 10
    nums, black = build_truth(n, rng)
    img = render(n, nums, black, pale=True)
    bbox = recognizer.find_board_bbox(img, cfg)
    if bbox is None or (bbox[2] - bbox[0]) < n * 20:
        return _fail(f"浅线盘 find_board_bbox 失败: {bbox}")
    got = recognizer.read_black_cells(img, cfg)
    if got is None:
        return _fail("浅线盘 read_black_cells 返回 None(放宽重试没接上)")
    gn, gcells = got
    if gn != n or gcells != black:
        return _fail(f"浅线盘回读不一致: n={gn}/{n} "
                     f"cells={len(gcells ^ black)} 差异")
    print(f"  [ok] 浅线盘锚点+回读一致 (n={gn}, 黑格 {len(gcells)} 全对)")
    return True


def case_noise_not_rescued(cfg, old):
    """无题噪声图: 放宽窗不许把它'救'成假棋盘."""
    rng = np.random.default_rng(17)
    arr = rng.integers(0, 256, size=(400, 400)).astype(np.uint8)
    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    err_new = None
    try:
        puz = recognizer.recognize(img, cfg, LOG)
        return _fail(f"噪声图被识别成棋盘 n={puz.n} — 放宽窗误救!")
    except recognizer.RecognitionError as e:
        err_new = str(e)
    if old is not None:
        try:
            old.recognize(img, cfg, LOG)
            return _fail("噪声图在旧代码下也被接受(测试本身无效?)")
        except Exception:
            pass
    print(f"  [ok] 噪声图仍被拒绝: {err_new[:40]}...")
    return True


def case_pale_boundary(cfg):
    """适应边界断言: 线芯 196(距格底 8 阶)必须救活; 200(距格底 4 阶,
    已进入格底自身抖动带)必须诚实拒绝 — 物理不可分区不许产出错盘."""
    rng = random.Random(23)
    n = 10
    nums, black = build_truth(n, rng)
    img = render(n, nums, black, pale=True, pale_core=196)
    try:
        new = recognizer.recognize(img, cfg, LOG)
    except recognizer.RecognitionError as e:
        return _fail(f"线芯196 应被放宽档救活, 却被拒: {e}")
    if new.n != n or dict(new.nums) != nums:
        return _fail(f"线芯196 盘识别错误: n={new.n}")
    if not any("放宽" in w for w in new.warnings):
        return _fail("线芯196 盘成功但未走放宽档(走的哪条路?)")
    print("  [ok] 线芯 196(距格底 8 阶) 被放宽档救活, 识别全对")
    img = render(n, nums, black, pale=True, pale_core=200)
    try:
        new = recognizer.recognize(img, cfg, LOG)
        return _fail(f"线芯200 物理上不可靠盘被认成 n={new.n} — "
                     f"错盘作答比拒绝更糟, 放宽策略需收紧")
    except recognizer.RecognitionError:
        print("  [ok] 线芯 200 盘被诚实拒绝(不产生错误作答)")
    return True


def main():
    cfg = load_cfg()
    old = load_old_recognizer()
    res = []
    print("[case 1] 普通深线盘 — 主路径与改造前逐位一致")
    res.append(case_normal_identical(cfg, old))
    print("[case 2] 浅线盘(双色反锯齿) — 旧必挂 / 新识别+求解")
    res.append(case_pale_rescued(cfg, old))
    print("[case 2b] 适应边界 — 线芯196必须救活 / 线芯200(物理不可靠)必须拒绝")
    res.append(case_pale_boundary(cfg))
    print("[case 3] 浅线盘 — 锚点定位与涂黑回读同步救活")
    res.append(case_pale_side_entries(cfg))
    print("[case 4] 噪声图 — 放宽窗不许误救")
    res.append(case_noise_not_rescued(cfg, old))
    mat = os.path.join(BASE, "测试素材", "浅线50x50_20260923.png")
    print("[case 5] 真实浅线截图素材(存在则端到端)")
    if os.path.exists(mat):
        img = cv2.imread(mat)
        try:
            puz = recognizer.recognize(img, cfg, LOG)
            st, grid, _ = sat_engine.solve_auto(puz.n, puz.nums,
                                                time_budget=60)
            okv, errs = solver.verify(puz.n, puz.nums, grid) if st == "ok" \
                else (False, [st])
            res.append(True if okv else _fail(f"素材端到端失败: {errs[:2]}"))
            print(f"  [ok] 真实素材: {puz.describe()} 端到端通过")
        except recognizer.RecognitionError as e:
            res.append(_fail(f"素材识别失败: {e}"))
    else:
        print("  [skip] 素材不在(把截图存为 测试素材/浅线50x50_20260923.png)")
        res.append(True)
    print(f"\n== 回归结果: {sum(res)}/{len(res)} 通过 ==")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
