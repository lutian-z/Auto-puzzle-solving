# -*- coding: utf-8 -*-
"""main.py — 高楼谜题 屏幕框选 / 截图 / 识别 / 求解 / 键鼠填答 主程序.

依赖: mss opencv-python numpy pyautogui
用法:
    python main.py            # 实机: 框选→识别→求解→填数→回车

目录: 与 solver.py、templates.npz 同目录; 模板数据文件随项目分发.
"""
import os
import sys
import time

import cv2
import numpy as np

import solver

IS_WINDOWS = sys.platform.startswith("win")

CONFIG = {
    "click_interval": 0.006,       # 点击格子到键入数字的间隔(秒)
    "cell_delay": 0.010,           # 每格填完后的停顿
    "jitter": 1.5,                 # 点击抖动像素
    "confidence": 0.55,            # 数字识别置信度阈值(低于则警告)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_NPZ_PATH = os.path.join(BASE_DIR, "templates.npz")
TEMPLATE_SIZE = 28

# ======================================================================
# 数字模板(随项目分发的数据文件)
# ======================================================================

def extract_glyph(ink):
    """把二值墨迹归一化到 TEMPLATE_SIZE×TEMPLATE_SIZE, 返回 float32 掩码."""
    if ink is None or not ink.any():
        return None
    g = (ink.astype(np.uint8)) * 255
    ys, xs = np.where(g)
    if len(ys) == 0:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = (TEMPLATE_SIZE - 8) / max(g.shape)
    nw = max(1, int(g.shape[1] * sc))
    nh = max(1, int(g.shape[0] * sc))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    y0c = (TEMPLATE_SIZE - nh) // 2
    x0c = (TEMPLATE_SIZE - nw) // 2
    canvas[y0c:y0c + nh, x0c:x0c + nw] = g
    return (canvas.astype(np.float32)) / 255.0


def _match_score(glyph_b, tpl_b):
    """交集/并集 IoU 分数. 入参均为 28×28 布尔掩码."""
    inter = (glyph_b & tpl_b).sum()
    union = (glyph_b | tpl_b).sum()
    return inter / union if union else 0.0


def match_digit(glyph, templates):
    """返回 (digit, best_score). 无匹配返回 (0, 0). 模板值为布尔掩码."""
    if glyph is None or not glyph.any():
        return 0, 0.0
    a = glyph > 0.5
    best_d, best_s = 0, 0.0
    for d, tpl in templates.items():
        s = _match_score(a, tpl)
        if s > best_s:
            best_s, best_d = s, d
    return best_d, best_s


# ======================================================================
# 几何检测: 定位棋盘格线、n、格子中心
# ======================================================================

def _line_positions(proj, thr):
    """投影数组上取超过阈值的连续段中心."""
    out = []
    i = 0
    n = len(proj)
    while i < n:
        if proj[i] >= thr:
            j = i
            while j + 1 < n and proj[j + 1] >= thr:
                j += 1
            out.append(int(round((i + j) / 2.0)))
            i = j + 1
        else:
            i += 1
    return out


def detect_bands(gray):
    """检测棋盘格线(双线风格或单线风格).

    返回 (n, centers, starts, ends, horizontal):
      centers/starts/ends 为数组, 均沿水平方向; horizontal 指示是沿行(水平线)还是列.
    失败返回 None.
    """
    h, w = gray.shape
    dark = (gray < 128).astype(np.uint8)
    v_counts = dark.sum(axis=0)   # 每列暗像素数 → 竖直格线
    h_counts = dark.sum(axis=1)   # 每行暗像素数 → 水平格线
    v_thr = max(40, h * 0.4)
    h_thr = max(40, w * 0.4)
    v_lines = _line_positions(v_counts, v_thr)
    h_lines = _line_positions(h_counts, h_thr)

    res_v = _bands_from_lines(v_lines)
    res_h = _bands_from_lines(h_lines)
    if res_v is None or res_h is None:
        return None
    n_v, c_v, s_v, e_v = res_v
    n_h, c_h, s_h, e_h = res_h
    if n_v != n_h:
        return None
    return n_v, c_v, s_v, e_v, c_h, s_h, e_h


def _bands_from_lines(lines):
    """从一条轴上的格线位置推出格子带.

    兼容三种风格: 相邻单线网格; 双线网格(每格一对线); 双线网格外加整幅外框线.
    返回 (n, centers, starts, ends) 或 None.
    """
    lines = [float(x) for x in sorted(set(lines))]
    if len(lines) < 2:
        return None

    def reg(widths):
        return float(widths.std() / widths.mean()) if widths.mean() > 0 else 99.0

    def make(pairs):
        starts = np.array([p[0] for p in pairs])
        ends = np.array([p[1] for p in pairs])
        widths = ends - starts
        if widths.min() <= 4:
            return None
        return len(starts), (starts + ends) / 2.0, starts, ends

    candidates = []
    # A: 相邻单线网格
    a = make(list(zip(lines[:-1], lines[1:])))
    if a is not None:
        candidates.append((reg(a[3] - a[2]), a[0], a))
    # B: 双线配对 offset0 (0,1),(2,3),...
    if len(lines) % 2 == 0:
        b = make(list(zip(lines[0::2], lines[1::2])))
        if b is not None:
            candidates.append((reg(b[3] - b[2]), b[0], b))
    # C: 双线配对 offset1 (1,2),(3,4),... (丢掉首尾外框线)
    if len(lines) % 2 == 0:
        c = make(list(zip(lines[1::2], lines[2::2])))
        if c is not None:
            candidates.append((reg(c[3] - c[2]), c[0], c))

    if not candidates:
        return None
    REG_MAX = 0.22
    good = [t for t in candidates if t[0] < REG_MAX]
    if not good:
        return None
    # 优先宽度均匀; 其次格子数多
    good.sort(key=lambda t: (t[0], -t[1]))
    return good[0][2]


def _clue_band(gray, side, board_edge):
    """返回该侧提示块带的跨度(像素区间). side: 'top'/'bottom'/'left'/'right'."""
    h, w = gray.shape
    nonwhite = gray < 235
    if side == "top":
        band = np.where(nonwhite[:board_edge, :].sum(axis=1) > 0)[0]
    elif side == "bottom":
        band = np.where(nonwhite[board_edge:, :].sum(axis=1) > 0)[0]
        if len(band):
            band = band + board_edge
    elif side == "left":
        band = np.where(nonwhite[:, :board_edge].sum(axis=0) > 0)[0]
    elif side == "right":
        band = np.where(nonwhite[:, board_edge:].sum(axis=0) > 0)[0]
        if len(band):
            band = band + board_edge
    if len(band) == 0:
        return None
    return int(band.min()), int(band.max())


def _crop_around(gray, cx, cy, half_w, half_h):
    h, w = gray.shape
    x0 = max(0, int(cx - half_w)); x1 = min(w, int(cx + half_w))
    y0 = max(0, int(cy - half_h)); y1 = min(h, int(cy + half_h))
    return gray[y0:y1, x0:x1]


def _extract_digit(gray_patch):
    """从格子/提示块patch中提取内部数字字形.

    用 Otsu 分离墨迹与背景(灰色填充视为背景), 再取"不碰patch边缘"且包围盒
    最小的连通组件 —— 即位于宫格/提示块内部的数字本体, 避开四周边框与填充.
    宫格裁剪应精确对应格子边界(边框恰好贴边), 空格无内部组件返回 None.
    """
    if gray_patch is None or gray_patch.size == 0:
        return None
    _, mask = cv2.threshold(gray_patch, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    h, w = gray_patch.shape
    m = max(1, int(min(h, w) * 0.04))
    best = None
    best_box = None
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        touches = (x <= m or y <= m or x + bw >= w - m or y + bh >= h - m)
        if not touches:
            box = bw * bh
            if best_box is None or box < best_box:
                best_box = box
                best = (labels == i).astype(np.uint8)
    if best is None:
        return None
    return extract_glyph(best)


# ======================================================================
# 识别一道题
# ======================================================================

def recognize_puzzle(img, templates):
    """识别一整道高楼题. 返回 (n, top, bottom, left, right, given, det) 或 None.

    top/bottom/left/right: 长度 n 的列表, 0=该侧无约束.
    given: n×n 列表, 0=空格.
    det: detect_bands 的几何结果(格线中心等), 供填数时换算格心复用.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = detect_bands(gray)
    if det is None:
        print("[识别] 未能定位棋盘格线")
        return None
    n, c_v, s_v, e_v, c_h, s_h, e_h = det
    h, w = gray.shape

    # 棋盘外边界
    top_edge = int(s_h[0]); bot_edge = int(e_h[-1])
    left_edge = int(s_v[0]); right_edge = int(e_v[-1])
    cell_w = float(np.median(e_v - s_v))
    cell_h = float(np.median(e_h - s_h))

    # 提示带
    top_band = _clue_band(gray, "top", top_edge)
    bot_band = _clue_band(gray, "bottom", bot_edge)
    left_band = _clue_band(gray, "left", left_edge)
    right_band = _clue_band(gray, "right", right_edge)

    # 宫格内已知数字
    given = [[0] * n for _ in range(n)]
    warn_cells = []
    for r in range(n):
        for c in range(n):
            patch = gray[int(s_h[r]):int(e_h[r]), int(s_v[c]):int(e_v[c])]
            glyph = _extract_digit(patch)
            d, score = match_digit(glyph, templates)
            if d and score >= CONFIG["confidence"]:
                given[r][c] = d
            elif glyph is not None and glyph.any() and score < CONFIG["confidence"]:
                warn_cells.append((r, c, d, round(score, 2)))
    if warn_cells:
        for r, c, d, sc in warn_cells[:20]:
            print(f"[警告] 宫内格({r+1},{c+1}) 数字置信度低: "
                  f"匹配 {d if d else '-'} 分 {sc} < {CONFIG['confidence']}")

    # 四周提示数字
    def read_side(centers, band, axis):
        # axis: 'x' 或 'y'; band: (min,max) 像素跨度; centers: 该侧每个提示的锚点
        out = []
        for ac in centers:
            if axis == "x":
                patch = _crop_around(gray, ac, (band[0] + band[1]) / 2.0,
                                     cell_w * 0.6, (band[1] - band[0]) * 0.6)
            else:
                patch = _crop_around(gray, (band[0] + band[1]) / 2.0, ac,
                                     (band[1] - band[0]) * 0.6, cell_h * 0.6)
            glyph = _extract_digit(patch)
            d, score = match_digit(glyph, templates)
            out.append(d if (d and score >= CONFIG["confidence"]) else 0)
        return out

    top = read_side(c_v, top_band, "x") if top_band else [0] * n
    bottom = read_side(c_v, bot_band, "x") if bot_band else [0] * n
    left = read_side(c_h, left_band, "y") if left_band else [0] * n
    right = read_side(c_h, right_band, "y") if right_band else [0] * n

    det = (n, c_v, s_v, e_v, c_h, s_h, e_h)
    return n, top, bottom, left, right, given, det


def print_puzzle(n, top, bottom, left, right, given):
    """控制台打印还原的题目, 便于人工核对."""
    def fmt(v):
        return " " if v == 0 else str(v)
    print("   " + " ".join(fmt(t) for t in top))
    for r in range(n):
        print(f"{fmt(left[r])}  " + " ".join(fmt(given[r][c]) for c in range(n)) +
              f"  {fmt(right[r])}")
    print("   " + " ".join(fmt(b) for b in bottom))


# ======================================================================
# 数字模板加载(templates.npz 与代码同目录分发)
# ======================================================================

def build_templates():
    """加载随项目分发的数字模板 templates.npz. 返回 {digit: 28x28 bool 掩码}.

    模板由开发期素材真实字形生成, 覆盖全部 1..9 字形(素材 1-4 缺失的 9 由
    2026-09-01 事故题真实样本补齐, 见交付说明); 缺失/损坏时报可读错误而非
    静默失败. 布尔化只做一次, 供 IoU 匹配直接使用.
    """
    try:
        data = np.load(TEMPLATE_NPZ_PATH, allow_pickle=False)
        tpls = {int(k): data[k].astype(np.float32) > 0.5 for k in data.files}
        if all(d in tpls for d in range(1, 10)):
            print(f"[模板] 从 {os.path.basename(TEMPLATE_NPZ_PATH)} 加载 "
                  f"{len(tpls)} 个模板")
            return tpls
    except Exception:
        pass
    raise SystemExit(
        "无法读取数字模板文件 templates.npz, 请确认它与 main.py "
        "位于同一目录且内容完整")


# ======================================================================
# 框选 / 截图 / 键鼠
# ======================================================================

def set_dpi_aware():
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _physical_scaling():
    import tkinter as tk
    import mss
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    try:
        tk_w = root.winfo_screenwidth()
        tk_h = root.winfo_screenheight()
        with mss.MSS() as sct:
            phys_w = sct.monitors[0]["width"]
            phys_h = sct.monitors[0]["height"]
        sx = (phys_w / tk_w) if tk_w else 1.0
        sy = (phys_h / tk_h) if tk_h else 1.0
    except Exception:
        sx = sy = 1.0
    root.destroy()
    return sx, sy


def select_region():
    """半透明全屏覆盖层, 用户从左上拖到右下框选游戏区. 返回物理像素 (x0,y0),(x1,y1)."""
    import tkinter as tk

    sx, sy = _physical_scaling()
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.3)
    root.configure(bg="black")
    root.wait_visibility(root)

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "rect": None, "done": False}

    def on_press(event):
        state["x0"], state["y0"] = event.x_root, event.y_root
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x_root, event.y_root, event.x_root, event.y_root,
            outline="red", width=2)

    def on_drag(event):
        if state["rect"]:
            canvas.coords(state["rect"], state["x0"], state["y0"],
                          event.x_root, event.y_root)
            w = abs(event.x_root - state["x0"]); h = abs(event.y_root - state["y0"])
            canvas.itemconfig(state["size_text"],
                              text=f"框选尺寸: {w} x {h}  (应覆盖整个棋盘, 可含四周提示)")
            canvas.coords(state["size_text"], 10, 60)

    def on_release(event):
        state["done"] = True
        state["x1"], state["y1"] = event.x_root, event.y_root

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<Return>", lambda e: root.destroy())

    canvas.create_text(10, 10, anchor="nw", fill="white",
                       font=("Microsoft YaHei", 16) if IS_WINDOWS else ("DejaVu Sans", 16),
                       text="拖动鼠标框选整个高楼棋盘(含四周提示)\n松开完成, ESC 取消")
    state["size_text"] = canvas.create_text(
        10, 60, anchor="nw", fill="yellow",
        font=("Microsoft YaHei", 13) if IS_WINDOWS else ("DejaVu Sans", 13),
        text="框选尺寸: -")

    while not state["done"]:
        root.update()
        time.sleep(0.02)
    root.destroy()

    x0, y0 = min(state["x0"], state["x1"]), min(state["y0"], state["y1"])
    x1, y1 = max(state["x0"], state["x1"]), max(state["y0"], state["y1"])
    if x1 - x0 < 10 or y1 - y0 < 10:
        sys.exit("框选区域过小, 退出")
    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        print(f"[DPI] tkinter 逻辑像素换算比例 x={sx:.2f} y={sy:.2f}")
    return (int(round(x0 * sx)), int(round(y0 * sy))), \
        (int(round(x1 * sx)), int(round(y1 * sy)))


_MSS = None   # mss 实例复用, 避免每次截屏重建


def grab_screen(bbox):
    import mss
    global _MSS
    left, top = bbox[0]
    right, bottom = bbox[1]
    monitor = {"left": left, "top": top,
               "width": right - left, "height": bottom - top}
    if _MSS is None:
        _MSS = mss.mss()
    try:
        shot = _MSS.grab(monitor)
    except Exception:
        _MSS = mss.mss()      # 显示配置变化等异常时重建一次
        shot = _MSS.grab(monitor)
    img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def compute_cell_centers(n, c_v, c_h, offset):
    """棋盘格中心换算到屏幕坐标."""
    ox, oy = offset
    return [[(int(round(ox + c_v[c])), int(round(oy + c_h[r])))
             for c in range(n)] for r in range(n)]


def fill_board(solution, centers, given):
    """点击+键入填数, 跳过初值格. 返回填入格数."""
    import pyautogui
    import random
    pyautogui.PAUSE = 0.0
    pyautogui.MINIMUM_DURATION = 0.0
    pyautogui.MINIMUM_SLEEP = 0.0

    n = len(solution)
    filled = 0
    for r in range(n):
        for c in range(n):
            if given[r][c]:
                continue
            cx, cy = centers[r][c]
            jx = random.uniform(-CONFIG["jitter"], CONFIG["jitter"])
            jy = random.uniform(-CONFIG["jitter"], CONFIG["jitter"])
            tx, ty = int(round(cx + jx)), int(round(cy + jy))
            # click 带坐标会瞬间定位(不播放平滑移动动画)并点击
            pyautogui.click(tx, ty)
            time.sleep(CONFIG["click_interval"])
            pyautogui.press(str(solution[r][c]))
            filled += 1
            time.sleep(CONFIG["cell_delay"])
    return filled


# ======================================================================
# 主流程
# ======================================================================

def run_pipeline(templates, bbox=None, img=None):
    """识别+求解+填数. bbox 为屏幕区域; 或直接给 img 做识别(不填数)."""
    if img is None:
        img = grab_screen(bbox)
    res = recognize_puzzle(img, templates)
    if res is None:
        print("[错误] 识别失败")
        return False
    n, top, bottom, left, right, given, det = res
    print(f"[识别] n={n}")
    print_puzzle(n, top, bottom, left, right, given)

    sol = solver.solve(n, top, bottom, left, right, given)
    if sol is None:
        print("[错误] 求解失败(无解或超时)")
        return False
    print("[求解] 完成")

    if img is not None and bbox is not None:
        n2, c_v, s_v, e_v, c_h, s_h, e_h = det
        centers = compute_cell_centers(n, c_v, c_h, bbox[0])
        filled = fill_board(sol, centers, given)
        print(f"[填数] 已填入 {filled} 格")
        # 填数后复检: 重新截屏逐格核对显示值与求解结果(可发现点击漂移/键入
        # 丢失等实机意外); 不一致则不提交回车, 交由人工处理.
        # 先短等后快查; 仅"部分不符"时才追加等待重查(可能是网站渲染未完成),
        # 全对/全空立即定论, 避免固定长等待.
        time.sleep(0.25)
        ok_fill, bad, miss = verify_fill(templates, bbox, n, sol, given)
        for _ in range(2):
            if ok_fill is None or not (bad or (miss and len(miss) < filled)):
                break
            time.sleep(0.25)
            ok_fill, bad, miss = verify_fill(templates, bbox, n, sol, given)
        if ok_fill is None:
            print("[警告] 填数后复检识别失败, 无法核对, 仍提交")
        elif bad:
            print(f"[安全] 复检发现 {len(bad)} 格显示值与求解结果不符, 不提交回车:")
            for r, c, want, got in bad[:10]:
                print(f"    ({r},{c}) 应为 {want}, 实际 {'空' if got == 0 else got}")
            return False
        elif miss and len(miss) < filled:
            print(f"[安全] 复检发现 {len(miss)} 格漏填, 不提交回车: "
                  + ", ".join(f"({r},{c})缺{v}" for r, c, v in miss[:10]))
            return False
        elif miss:
            # 所有填入格都没显示 —— 网站可能是提交后才渲染, 跳过核对
            print("[警告] 复检未看到任何填入内容(网站可能不实时显示), 仍提交")
        else:
            print("[复检] 填入内容与求解结果一致")
        import pyautogui
        pyautogui.press("enter")
        print("[提交] 已按回车")
    return True


def verify_fill(templates, bbox, n, sol, given):
    """填数后重新截屏逐格核对. 返回 (ok_fill, bad, miss).

    ok_fill: True=正常核对; None=复检识别失败(无法核对).
    bad: [(r,c,期望,实际)] 显示值与解不符的格(含网站拒绝输入的原始给定).
    miss: [(r,c,期望)] 我们填过但屏幕上仍为空的格.
    """
    img2 = grab_screen(bbox)
    res2 = recognize_puzzle(img2, templates)
    if res2 is None:
        return None, [], []
    _, _t2, _b2, _l2, _r2, g2, _det2 = res2
    bad = [(r + 1, c + 1, sol[r][c], g2[r][c])
           for r in range(n) for c in range(n)
           if g2[r][c] and g2[r][c] != sol[r][c]]
    miss = [(r + 1, c + 1, sol[r][c])
            for r in range(n) for c in range(n)
            if not given[r][c] and not g2[r][c]]
    return True, bad, miss


def main():
    set_dpi_aware()
    templates = build_templates()
    print("[流程] 请在屏幕上拖动框选整个高楼棋盘(含四周提示)...")
    p0, p1 = select_region()
    print(f"[流程] 框选区域: {p0} -> {p1}")
    img = grab_screen((p0, p1))
    run_pipeline(templates, bbox=(p0, p1), img=img)


if __name__ == "__main__":
    main()
