# -*- coding: utf-8 -*-
"""ms_auto.py — 扫雷自动化主库.

由原 ms_common(桌面自动化) + ms_grid(网格检测/图像对齐) +
ms_recognize(格子分类) + ms_auto(滚动拼接识别/滚动回填) 合并而成。
主程序 main.py 与本模块自身都从这里导入全部能力。

关键原则: 滚动偏移量由图像对齐(互相关+网格线精修)计算, 绝不依赖滚轮格数;
识别必须确认真实底边后才允许求解; 回填按整行完成且不与原题整屏强匹配。
"""
import sys
import time
import threading

import cv2
import numpy as np


# ======================================================================
# 一、桌面自动化层(原 ms_common.py)
# ======================================================================


IS_WINDOWS = sys.platform.startswith("win")


class StopFlag:
    """注册一个热键(默认 ESC), 触发后 .stopped 为 True."""
    def __init__(self):
        self._stop = threading.Event()
        self._hook = None

    def start(self, hotkey="esc"):
        import keyboard
        keyboard.add_hotkey(hotkey, self._stop.set)
        print(f"[热键] 已注册中断热键 {hotkey.upper()}")

    @property
    def stopped(self):
        return self._stop.is_set()

    def reset(self):
        self._stop.clear()


def set_dpi_aware():
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ----------------------------------------------------------------------
# 区域选择(拖动框选 / 点击两个点, 按鼠标手势自动识别)
# ----------------------------------------------------------------------

MOVE_THRESHOLD = 8   # 按下后移动超过该逻辑像素判定为拖动框选, 否则视为单击
MIN_BOX_SIZE = 10    # 有效框选的最小边长(逻辑像素), 过小提示重来


def select_region():
    """全屏覆盖层, 两种互相独立的选题方式, 按鼠标手势自动判断:

      - 拖动框选: 按下左键并移动超过阈值后松开, 框住完整题目.
        题目所有信息都在框内, 走独立管线: 单屏识别 + 框内直接插旗,
        全程不滚动. 返回 ("box", (left, top, right, bottom)).
      - 点击两个点: 按下后原地松开(移动未超阈值)视为单击, 等待第二下.
        第一个点为左边界点, 第二个点为右边界点, 两点中较高纵坐标为顶边.
        点满两点自动确认. 适用超出屏幕的超大棋盘, 识别与回填会滚动拼接.
        返回 ("points", (left, top, right)).

    两种方式可随时切换: 已记录过点击点后再拖动, 则放弃点击点改用框选.
    按 ESC 取消. 坐标均为物理像素。
    """
    return _select_region_tk()


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


def _select_region_tk():
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

    state = {
        "pts": [],         # 点击两点模式已记录的点(逻辑像素)
        "markers": [],     # 点击点的圆圈标记
        "rect": None,      # 拖动框选的实时矩形
        "press": None,     # 本次按下的起点(逻辑像素)
        "dragging": False, # 本次按下是否已升级为拖动框选
        "done": False,
        "result": None,    # ("box", (x0, y0, x1, y1)) / ("points", [(x,y),(x,y)])
    }

    def set_hint(text):
        canvas.itemconfig(state["hint"], text=text)

    def clear_points():
        for m in state["markers"]:
            canvas.delete(m)
        state["markers"] = []
        state["pts"] = []

    def clear_rect():
        if state["rect"]:
            canvas.delete(state["rect"])
            state["rect"] = None

    def on_press(event):
        state["press"] = (event.x_root, event.y_root)
        state["dragging"] = False

    def on_drag(event):
        press = state["press"]
        if press is None:
            return
        if not state["dragging"]:
            if max(abs(event.x_root - press[0]),
                   abs(event.y_root - press[1])) < MOVE_THRESHOLD:
                return
            # 移动超阈值: 升级为拖动框选, 放弃已记录的点击点
            state["dragging"] = True
            clear_points()
            clear_rect()
            state["rect"] = canvas.create_rectangle(
                press[0], press[1], event.x_root, event.y_root,
                outline="red", width=2)
        if state["rect"]:
            canvas.coords(state["rect"], press[0], press[1],
                          event.x_root, event.y_root)
            w = abs(event.x_root - press[0])
            h = abs(event.y_root - press[1])
            canvas.itemconfig(state["size_text"],
                              text=f"框选尺寸: {w} x {h}  "
                                   "(框住整个棋盘即可, 可略大于棋盘)")
            canvas.coords(state["size_text"], 10, 60)

    def on_release(event):
        press = state["press"]
        state["press"] = None
        if press is None:
            return
        if state["dragging"]:
            state["dragging"] = False
            x0 = min(press[0], event.x_root)
            y0 = min(press[1], event.y_root)
            x1 = max(press[0], event.x_root)
            y1 = max(press[1], event.y_root)
            if x1 - x0 < MIN_BOX_SIZE or y1 - y0 < MIN_BOX_SIZE:
                # 过小的"框"基本是手抖, 提示重来, 不当作框选也不计入点击点
                clear_rect()
                set_hint("框选区域过小, 请重新拖动框住整个棋盘\n"
                         "拖动框选, 或点击两个点(左点=左边界, 右点=右边界)")
                return
            state["result"] = ("box", (x0, y0, x1, y1))
            state["done"] = True
            return
        # 原地松开: 单击, 进入/继续点击两点模式
        state["pts"].append((event.x_root, event.y_root))
        state["markers"].append(canvas.create_oval(
            event.x_root - 6, event.y_root - 6,
            event.x_root + 6, event.y_root + 6,
            outline="red", width=3))
        if len(state["pts"]) == 1:
            set_hint("已记录第 1 点(左/右边界), 点击第 2 点确认\n"
                     "也可随时改为拖动框选整个棋盘")
        elif len(state["pts"]) == 2:
            x0 = min(state["pts"][0][0], state["pts"][1][0])
            x1 = max(state["pts"][0][0], state["pts"][1][0])
            y0 = min(state["pts"][0][1], state["pts"][1][1])
            set_hint(f"左边界 {x0}, 右边界 {x1}, 顶边 {y0}")
            state["result"] = ("points", list(state["pts"]))
            state["done"] = True

    def on_enter(event):
        # 回车兜底: 仅在点满两点时确认
        if len(state["pts"]) == 2:
            state["result"] = ("points", list(state["pts"]))
            state["done"] = True

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: (root.destroy(), sys.exit(0)))
    root.bind("<Return>", on_enter)

    state["hint"] = canvas.create_text(
        10, 10, anchor="nw", fill="white", font=("Microsoft YaHei", 16),
        text="拖动框选完整题目(信息都在框内, 全程不滚动), 或点击两个点\n"
             "(左点=左边界, 右点=右边界, 超大棋盘用, 会滚动拼接)\n"
             "松开/点满两点自动开始, ESC 取消")
    state["size_text"] = canvas.create_text(
        10, 90, anchor="nw", fill="yellow", font=("Microsoft YaHei", 13),
        text="框选尺寸: -")

    while not state["done"]:
        root.update()
        time.sleep(0.02)
    root.destroy()

    if state["result"] is None:
        sys.exit("未完成选题, 退出")

    kind, data = state["result"]
    if kind == "box":
        x0, y0, x1, y1 = data
        left, right, top = x0, x1, y0
        bottom = y1
        left = int(round(left * sx))
        right = int(round(right * sx))
        top = int(round(top * sy))
        bottom = int(round(bottom * sy))
        return "box", (left, top, right, bottom)
    (ax, ay), (bx, by) = data
    left, right, top = min(ax, bx), max(ax, bx), min(ay, by)
    left = int(round(left * sx))
    right = int(round(right * sx))
    top = int(round(top * sy))
    return "points", (left, top, right)


# ----------------------------------------------------------------------
# 截屏
# ----------------------------------------------------------------------

def grab_screen(bbox):
    """按物理像素边界截屏, 返回 BGR ndarray.

    bbox = (left, top, right, bottom) 屏幕坐标.
    """
    import cv2
    import mss
    import numpy as np
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None
    monitor = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }
    with mss.MSS() as sct:
        shot = sct.grab(monitor)
    img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
        shot.height, shot.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_screen_size():
    import mss
    with mss.MSS() as sct:
        m = sct.monitors[0]
    return m["width"], m["height"]


# ----------------------------------------------------------------------
# 鼠标操作
# ----------------------------------------------------------------------

# pyautogui 默认在每次动作之间 PAUSE 0.1 秒，对几百次右键回填是巨大的浪费。
# 这里统一把 PAUSE 压到 0（动作本身仍由操作系统正常投递，只是不再人为睡 0.1s）。
_PG_PAUSE_SET = False


def _pg():
    global _PG_PAUSE_SET
    import pyautogui
    if not _PG_PAUSE_SET:
        pyautogui.PAUSE = 0.0
        pyautogui.MINIMUM_DURATION = 0.0
        pyautogui.MINIMUM_SLEEP = 0.0
        _PG_PAUSE_SET = True
    return pyautogui


def move_mouse(x, y):
    # duration=0: 瞬间定位, 不播放平滑移动动画
    _pg().moveTo(int(x), int(y), duration=0)


def scroll_wheel(clicks):
    """滚轮滚动. clicks>0 向上滚, <0 向下滚.

    滚轮作用于当前鼠标所在处。整个流程里鼠标始终停在棋盘上(上一动作是
    右键插旗或点选棋盘), 原地滚动即可, 无需先移到屏幕/棋盘中心。
    """
    _pg().scroll(int(clicks))


def right_click(x, y, jitter=1.5):
    """在 (x,y) 附近右键点击(标记雷), 带随机抖动."""
    import random
    jx = random.uniform(-jitter, jitter)
    jy = random.uniform(-jitter, jitter)
    tx, ty = int(round(x + jx)), int(round(y + jy))
    _pg().click(x=tx, y=ty, button="right")


def press_enter():
    _pg().press("enter")




# ======================================================================
# 二、网格检测与图像对齐(原 ms_grid.py)
# ======================================================================

DARK_GRAY = 130          # 隔线(深灰)阈值
SEP_THRESH = 0.50        # 某行/列被视为隔线所需暗像素占比
MIN_LINES = 3            # 至少检测到几条线才算有效

# ----------------------------------------------------------------------
# 隔线检测
# ----------------------------------------------------------------------

def _cluster_bands(indices, gap=4):
    """把连续索引聚成若干带, 返回每个带的中心位置(整数).

    gap 为允许的带内空隙; 一个"带"代表一条隔线的所有暗像素行/列,
    其中心即网格线所在位置.
    """
    if len(indices) == 0:
        return []
    bands = []
    start = prev = indices[0]
    for v in indices[1:]:
        if v - prev <= gap:
            prev = v
        else:
            bands.append(int(round((start + prev) / 2.0)))
            start = prev = v
    bands.append(int(round((start + prev) / 2.0)))
    return bands


def detect_h_bands(gray, thresh=SEP_THRESH):
    """检测横向隔线(整行深色占比高), 返回行位置列表."""
    h, w = gray.shape
    dark = (gray < DARK_GRAY).astype(np.uint8)
    frac = dark.sum(axis=1) / float(w)
    return _cluster_bands(np.where(frac > thresh)[0])


def detect_v_bands(gray, row_range=None, thresh=SEP_THRESH):
    """检测纵向隔线.

    row_range 指定统计用的行区间(棋盘纵向范围), 排除棋盘上下的背景,
    使纵向隔线检测更稳健.
    """
    h, w = gray.shape
    dark = (gray < DARK_GRAY).astype(np.uint8)
    if row_range is not None:
        y0, y1 = row_range
        y0, y1 = max(0, y0), min(h, y1)
        if y1 - y0 < 4:
            return []
        frac = dark[y0:y1, :].sum(axis=0) / float(y1 - y0)
    else:
        frac = dark.sum(axis=0) / float(h)
    return _cluster_bands(np.where(frac > thresh)[0])


def compute_pitch(positions):
    """隔线间距的中位数."""
    if positions is None or len(positions) < 2:
        return None
    diffs = np.diff(positions)
    return float(np.median(diffs))


def _filter_band_spacing(positions):
    """剔除间距异常的隔线带.

    棋盘的隔线间距应基本一致. 若检测到两条间距远小于主体间距的带(如棋盘
    边缘残带、界面元素), 保留主体间距更大的那条, 剔除异常者. 反复过滤直到
    间距一致.
    """
    if positions is None or len(positions) < 4:
        return list(positions)
    positions = list(positions)
    for _ in range(8):
        if len(positions) < 4:
            break
        diffs = np.array([positions[i + 1] - positions[i] for i in range(len(positions) - 1)])
        med = float(np.median(diffs))
        # 找到最小间距
        min_i = int(np.argmin(diffs))
        min_d = diffs[min_i]
        # 若最小间距远小于中位数(比如 < 0.6 中位数), 存在异常带
        if min_d < 0.6 * med:
            # 异常带相邻的两个间距: diffs[min_i] 与 diffs[min_i+1] (若存在)
            # 判断删除 positions[min_i] 还是 positions[min_i+1]
            # 看删除哪个能让相邻间距更接近 med
            left_d = diffs[min_i - 1] if min_i > 0 else None
            right_d = diffs[min_i + 1] if min_i + 1 < len(diffs) else None
            if left_d is None:
                # 左端出现“图像边缘 + 第一条真实网格线”时，应删除更靠外
                # 的图像边缘，而不是删掉第一条真实线。
                del positions[min_i]
            elif right_d is None:
                # 右端同理：保留倒数第二条真实网格线，删除贴近它的图像边缘。
                del positions[min_i + 1]
            else:
                # 比较: 删左 -> 新间距 = min_d + left_d; 删右 -> 新间距 = min_d + right_d
                # 选使新间距更接近 med 的
                if abs((min_d + left_d) - med) <= abs((min_d + right_d) - med):
                    del positions[min_i]
                else:
                    del positions[min_i + 1]
            continue
        break
    return positions


def _has_board_columns(gray, y0, y1):
    """区间 [y0, y1) 内是否仍有棋盘纵向网格结构.

    用于区分两类大间隔:
      - 漏检了若干行的真实棋盘线: 间隔区域仍是棋盘, 有规则、密集的纵向列线,
        且亮度处于棋盘格范围;
      - 棋盘下方的孤立深色线(浏览器窗口下沿/任务栏/桌面壁纸): 通常极暗或
        极亮, 且没有棋盘的列线密度。
    """
    if gray is None or y1 - y0 < 4:
        return False
    sub = gray[y0:y1, :]
    mean = float(sub.mean())
    # 棋盘格区域不是极暗或极亮的纯色块.
    if mean < 80 or mean > 235:
        return False
    width = gray.shape[1]
    vx = detect_v_bands(gray, row_range=(y0, y1))
    if len(vx) < MIN_LINES:
        return False
    gaps = np.diff(np.asarray(vx, dtype=float))
    if len(gaps) < 2:
        return False
    med = float(np.median(gaps))
    if med < 3:
        return False
    # 棋盘列线密度: (线数-1)*间距 ≈ 区域宽度(约 1 线/格). 桌面/壁纸上的
    # 零星竖线不满足该密度.
    density = (len(vx) - 1) * med / float(width)
    return density >= 0.35


def _trim_edge_outliers(positions, pitch, gray=None, max_ratio=1.5):
    """裁掉与棋盘网格不连续的首/尾孤立线带.

    浏览器窗口下沿、任务栏、桌面壁纸等会在棋盘下方(或上方)额外检测出一条
    深色横线。它与棋盘最后一条线的间距明显大于一格(> max_ratio*pitch),
    会被 count_grid_intervals 误算成多行, 也会被误当成“真实底边”。这里从
    两端裁掉这种孤立带。

    但若大间隔其实是“漏检了多行”造成的(间隔区域仍是棋盘, 有纵向列线),
    则保留尾部带, 由 count_grid_intervals 把大间隔还原成多行。

    安全护栏: 至少保留 MIN_LINES 条带, 避免把整个棋盘裁没。
    """
    if positions is None or len(positions) < 3 or pitch <= 0:
        return list(positions) if positions else []
    bands = sorted(set(int(round(v)) for v in positions))
    limit = max_ratio * float(pitch)
    while len(bands) > MIN_LINES:
        tail_gap = bands[-1] - bands[-2]
        head_gap = bands[1] - bands[0]
        if tail_gap > limit and not _has_board_columns(gray, bands[-2], bands[-1]):
            bands.pop()
            continue
        if head_gap > limit and not _has_board_columns(gray, bands[0], bands[1]):
            bands.pop(0)
            continue
        break
    return bands


def count_grid_intervals(positions, pitch):
    """根据相邻检测线之间跨过的格数统计总格数。

    浏览器缩放会把理论上的非整数格宽分配成 40/41/42 等不同像素宽度，
    因而不能用 ``总跨度 / 中位数间距`` 再四舍五入；40 格可能因此变成
    39 格。逐段计数既保留这些正常抖动，也能在漏掉一条线时把约 2*pitch
    的大间隔识别成两个格子。
    """
    if positions is None or len(positions) < 2 or pitch <= 0:
        return 0
    total = 0
    for left, right in zip(positions, positions[1:]):
        gap = float(right - left)
        total += max(1, int(round(gap / float(pitch))))
    return total


def regularize_lines(positions, pitch, n, lo=0, hi=None):
    """生成 n+1 条规整网格线, 尽量贴近检测到的隔线.

    检测到的真实线优先原样保留。只有相邻两条检测线的间隔跨过多个格子
    时，才在这两个真实端点之间做局部插值补线。不能从某个起点按固定
    ``pitch`` 一路外推，否则浏览器缩放造成的 40/42px 宽度分配会逐列
    累积相位误差，最终把单元格切到相邻列。
    """
    if positions is None or len(positions) == 0 or pitch <= 0:
        return None

    positions = sorted(set(int(round(v)) for v in positions))
    out = [positions[0]]
    for left, right in zip(positions, positions[1:]):
        steps = max(1, int(round((right - left) / float(pitch))))
        for step in range(1, steps + 1):
            # 以两条真实检测线为端点局部均分；最后一点严格等于 right。
            value = left + (right - left) * step / float(steps)
            out.append(int(round(value)))

    expected = n + 1
    if len(out) < expected:
        # 极端情况下末端线不可见但调用方已有更可靠的 n，才允许从最后
        # 一段的实际平均格宽向外补齐；正常 analyze_screen 不会走到这里。
        actual_pitch = ((out[-1] - out[0]) / float(max(1, len(out) - 1))
                        if len(out) > 1 else pitch)
        while len(out) < expected:
            out.append(int(round(out[-1] + actual_pitch)))
    elif len(out) > expected:
        out = out[:expected]
    if hi is not None:
        out = [min(max(v, lo), hi) for v in out]
    return out


def _band_thickness(gray, center, axis="h", width=8):
    """测量某条隔线的厚度(暗带宽度), 用于识别顶部/底部边框."""
    h, w = gray.shape
    half = width
    if axis == "h":
        y0, y1 = max(0, center - half), min(h, center + half + 1)
        seg = gray[y0:y1, max(0, w // 4):w - w // 4]
        dark_rows = [(seg[i] < DARK_GRAY).mean() > 0.5 for i in range(seg.shape[0])]
    else:
        x0, x1 = max(0, center - half), min(w, center + half + 1)
        seg = gray[max(0, h // 4):h - h // 4, x0:x1]
        dark_rows = [(seg[:, i] < DARK_GRAY).mean() > 0.5 for i in range(seg.shape[1])]
    # 找含 center 的连续暗带长度
    best = 0
    for i in range(len(dark_rows)):
        if dark_rows[i]:
            j = i
            while j + 1 < len(dark_rows) and dark_rows[j + 1]:
                j += 1
            if i <= half <= j:      # 包含中心
                best = max(best, j - i + 1)
            i = j + 1
    return best


def analyze_screen(img_bgr):
    """分析一屏画面, 返回网格结构信息 dict; 失败返回 None.

    返回:
      n       列数(棋盘尺寸, 方形)
      m       本屏可见行数
      vx      规整后的纵向网格线 (n+1 个)
      hy      规整后的横向网格线 (m+1 个)
      pitch   格子间距(像素)
      hy_raw  检测到的横向隔线(用于滚动对齐)
      vx_raw  检测到的纵向隔线
      top_is_border  首条横线是否为顶部边框(更厚)
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return None

    # 1. 先找横向隔线(整行暗)
    hy_raw = detect_h_bands(gray)
    if len(hy_raw) < MIN_LINES:
        return None

    # 2. 在棋盘纵向范围内统计纵向隔线
    row_lo, row_hi = hy_raw[0], hy_raw[-1]
    vx_raw = detect_v_bands(gray, row_range=(row_lo, row_hi))
    if len(vx_raw) < MIN_LINES:
        return None

    # 3. 间距(先剔除异常近的列带, 避免边界干扰)
    # 列带的真实间距应基本一致(约等于 pitch); 异常近的带(如边缘残带)会拉低中位数
    vx_raw = _filter_band_spacing(vx_raw)
    hy_raw = _filter_band_spacing(hy_raw)
    if len(vx_raw) < MIN_LINES or len(hy_raw) < MIN_LINES:
        return None

    pitch_h = compute_pitch(hy_raw)
    pitch_v = compute_pitch(vx_raw)
    if pitch_h is None or pitch_v is None:
        return None
    pitch = float(np.median([pitch_h, pitch_v]))
    if pitch < 3:
        return None

    # 3.5 裁掉与棋盘网格不连续的首/尾孤立横线(浏览器窗口下沿/任务栏/桌面
    # 壁纸上的深色线)。否则大间隔会被 count_grid_intervals 误算成多行, 也
    # 会被误当成“真实底边”, 导致“首屏可见 9 行”之类错误。
    hy_raw = _trim_edge_outliers(hy_raw, pitch, gray=gray)
    if len(hy_raw) < MIN_LINES:
        return None
    # 用裁剪后的棋盘纵向范围重新检测列线, 避免下方桌面/窗口线污染列线统计。
    row_lo, row_hi = hy_raw[0], hy_raw[-1]
    vx_raw = detect_v_bands(gray, row_range=(row_lo, row_hi))
    vx_raw = _filter_band_spacing(vx_raw)
    if len(vx_raw) < MIN_LINES:
        return None
    pitch_h = compute_pitch(hy_raw)
    pitch_v = compute_pitch(vx_raw)
    if pitch_h is None or pitch_v is None:
        return None
    pitch = float(np.median([pitch_h, pitch_v]))
    if pitch < 3:
        return None

    # 4. 列数(方形棋盘, 行数目标=列数)。逐间隔统计，避免缩放产生的
    # 交替像素宽度使 40 格被“总跨度/中位数”误算为 39 格。
    n = count_grid_intervals(vx_raw, pitch)
    if n < 1 or n > 200:
        return None
    n_count = len(vx_raw) - 1
    if n_count != n:
        pass  # 允许漏检, 以跨度推算为准(打印警告由调用方处理)

    # 5. 可见完整行数，同样逐间隔统计。
    m = count_grid_intervals(hy_raw, pitch)
    if m < 1 or m > 200:
        return None

    # 6. 规整网格线
    vx = regularize_lines(vx_raw, pitch, n, 0, w - 1)
    hy = regularize_lines(hy_raw, pitch, m, 0, h - 1)
    if vx is None or hy is None:
        return None

    # 7. 棋盘外边框判断(比普通内部隔线更厚)。不仅记录顶部，还记录
    # 所有可能的底边及它在当前屏之前跨过的完整行数。网页视口底沿有时
    # 也会形成一条深色横线；要求候选线下方仍留有少量截图空间，避免把
    # 屏幕边缘误当成棋盘底边。
    line_thickness = [_band_thickness(gray, p, "h") for p in hy_raw]
    top_thick = line_thickness[0]
    med_thick = float(np.median(line_thickness[1:-1] or [top_thick]))
    # 边框比普通隔线明显更厚(宽松阈值, 素材中边框9-12px, 隔线6px)
    top_is_border = top_thick >= med_thick + 2
    border_threshold = max(3.0, med_thick + 1.5)
    min_bottom_margin = max(4.0, pitch * 0.15)
    bottom_border_rows = []
    bottom_border_details = []
    rejected_bottom_borders = []
    for i in range(1, len(hy_raw)):
        margin_below = (h - 1) - hy_raw[i]
        if line_thickness[i] >= border_threshold:
            gap = float(hy_raw[i] - hy_raw[i - 1])
            crossed = max(1, int(round(gap / pitch)))
            per_row_gap = gap / crossed
            full_ratio = per_row_gap / pitch
            local_rows = count_grid_intervals(hy_raw[:i + 1], pitch)
            detail = {
                "local_rows": local_rows,
                "y": int(hy_raw[i]),
                "thickness": int(line_thickness[i]),
                "margin_below": float(margin_below),
                "last_gap": gap,
                "full_ratio": float(full_ratio),
            }
            # 真底边前的一格必须具有接近完整 pitch 的高度。视口底部或
            # IDE/浏览器横线常会把只露出半格的区域封成“伪末行”；仅凭
            # 线条较粗和下方有余量不足以证明最后一行已经看全。
            if (margin_below >= min_bottom_margin and
                    full_ratio >= 0.84):
                bottom_border_rows.append(local_rows)
                bottom_border_details.append(detail)
            else:
                rejected_bottom_borders.append(detail)

    return {
        "n": n,
        "m": m,
        "vx": vx,
        "hy": hy,
        "pitch": pitch,
        "hy_raw": hy_raw,
        "vx_raw": vx_raw,
        "top_is_border": top_is_border,
        "top_thick": top_thick,
        "med_thick": med_thick,
        "line_thickness": line_thickness,
        "bottom_border_rows": bottom_border_rows,
        "bottom_border_details": bottom_border_details,
        "rejected_bottom_borders": rejected_bottom_borders,
        "bottom_is_border": m in bottom_border_rows,
    }


# ----------------------------------------------------------------------
# 滚动偏移计算(图像对齐)
# ----------------------------------------------------------------------

def match_scroll_delta(prev_hy, cur_hy, pitch, bin_width=2.0):
    """估算两屏之间的垂直滚动偏移像素.

    依据: 相邻两屏必然有一部分网格线重合, 相同网格线在两屏中的 y 坐标差
    即滚动偏移. 对全部"可能匹配"的线对统计差值直方图, 取峰值.

    返回 (delta, score):
      delta  滚动偏移像素(>0 表示画面下移/向上滚; 通常 >0)
      score  匹配到的线对数(越大越可信)
    """
    if prev_hy is None or cur_hy is None or len(prev_hy) < 1 or len(cur_hy) < 1:
        return 0.0, 0
    deltas = []
    for py in prev_hy:
        for cy in cur_hy:
            deltas.append(float(py - cy))
    if not deltas:
        return 0.0, 0
    lo, hi = min(deltas), max(deltas)
    if hi - lo < 1e-6:
        return deltas[0], len(deltas)

    bins = int(np.ceil((hi - lo) / bin_width)) + 1
    hist, edges = np.histogram(deltas, bins=bins, range=(lo, hi))
    best_bin = int(np.argmax(hist))
    bin_lo, bin_hi = edges[best_bin], edges[best_bin + 1]
    cluster = [d for d in deltas if bin_lo <= d < bin_hi]
    delta = float(np.mean(cluster)) if cluster else (bin_lo + bin_hi) / 2.0
    score = hist[best_bin]
    return delta, score




# ----------------------------------------------------------------------
# 基于内容的图像对齐(滚动偏移)
# ----------------------------------------------------------------------



def align_screens(prev_bgr, cur_bgr, grid_prev, grid_cur, pitch,
                  max_shift_ratio=1.5):
    """基于内容的垂直滚动偏移估计(多模板匹配法).

    思路: 滚动距离不一定是格子间距整数倍, 两屏网格线会错位. 从 prev 的多个
    垂直位置取"模板带", 在 cur 中做模板匹配, 取匹配分最高的那个, 其 y 位移
    即为滚动偏移.

    向下滚动时 prev 的下部会保留在画面内, 因此模板带从低到高覆盖全屏,
    且底部模板带优先参与.

    返回 (delta, score):
      delta>0 表示 cur 内容相对 prev 上移 delta(即向下滚动).
    """
    if prev_bgr is None or cur_bgr is None:
        return 0.0, -1
    gp = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    hp, wp = gp.shape
    hc, wc = gc.shape
    if hp < 40 or hc < 40:
        return 0.0, -1

    # 棋盘区域(水平范围)
    def board_xrange(grid, w):
        if grid and (grid.get("vx") or grid.get("vx_raw")):
            vx = grid.get("vx") or grid["vx_raw"]
            x0, x1 = int(vx[0]), int(vx[-1])
            x0, x1 = max(0, x0), min(w, x1)
            if x1 - x0 > 8:
                return x0, x1
        return 0, w

    px0, px1 = board_xrange(grid_prev, wp)
    cx0, cx1 = board_xrange(grid_cur, wc)

    # 模板宽度: 棋盘中间 50%
    tw = int((px1 - px0) * 0.5)
    if tw < 8:
        tw = px1 - px0
    tx0 = px0 + (px1 - px0 - tw) // 2

    # 棋盘在 prev 图像内的纵向范围(用检测到的网格线 hy 界定)。这是滚动对齐
    # 能否成功的关键: 当浏览器窗口很小时, bbox 从用户顶边一直延伸到屏幕底部,
    # 其中很大一部分是棋盘下方的桌面/任务栏。若按整个 bbox 高度按比例放模板带,
    # 大多数带会落在棋盘之外(桌面), 内容匹配必然失败。用棋盘 hy 界定后, 无论
    # 可见多少行(7 行还是 14 行), 模板带都落在棋盘内部。
    b_y0, b_y1 = 0, hp
    if grid_prev is not None:
        _hy = grid_prev.get("hy") or grid_prev.get("hy_raw")
        if _hy is not None and len(_hy) >= 2:
            b_y0 = max(0, int(_hy[0]))
            b_y1 = min(hp, int(_hy[-1]))
    if b_y1 - b_y0 < 30:
        b_y0, b_y1 = 0, hp
    bh = b_y1 - b_y0

    # 模板带高度: 棋盘高度的 20%(足够覆盖一两行内容, 又不会太平滑).
    th = int(bh * 0.20)
    if th < 8:
        th = max(8, hp // 3)
    if th > bh // 2:
        th = max(8, bh // 2)

    # 降采样加速
    scale = min(1.0, 700.0 / max(hp, hc))

    # 模板带位置: 均匀分布在棋盘纵向范围内(不依赖可见行数), 下移方向稍密.
    tpl_positions = []
    for i in range(4):
        frac = (i + 0.5) / 4.0
        y = int(b_y0 + bh * frac - th / 2)
        if 0 <= y <= hp - th:
            tpl_positions.append(y)
    if not tpl_positions:
        y = int(b_y0 + bh // 2 - th // 2)
        if 0 <= y <= hp - th:
            tpl_positions.append(y)
    if not tpl_positions:
        tpl_positions = [max(0, min(hp - th, hp // 3))]

    best = None
    for ty0 in tpl_positions:
        tpl = gp[ty0:ty0 + th, tx0:tx0 + tw]
        if tpl.size == 0:
            continue
        if scale < 1.0:
            tpl_s = cv2.resize(tpl, (max(1, int(tw * scale)), max(1, int(th * scale))),
                               interpolation=cv2.INTER_AREA)
            gc_s = cv2.resize(gc, (max(1, int(wc * scale)), max(1, int(hc * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            tpl_s = tpl
            gc_s = gc
        if tpl_s.shape[0] >= gc_s.shape[0] or tpl_s.shape[1] >= gc_s.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(gc_s, tpl_s, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        _, mv, _, mloc = cv2.minMaxLoc(res)
        best_y = mloc[1] / scale
        d = ty0 - best_y
        if best is None or mv > best[0]:
            best = (float(mv), d)

    if best is None:
        return 0.0, -1
    return best[1], best[0]





# ======================================================================
# 三、格子分类(原 ms_recognize.py)
# ======================================================================

# BGR 参考色(OpenCV 读图)
COLOR_REF = {
    1: (255, 0, 0),      # 蓝
    2: (0, 153, 0),      # 绿
    3: (0, 0, 255),      # 红
    4: (153, 0, 0),      # 深蓝
    5: (0, 0, 153),      # 暗红
    6: (153, 153, 0),    # 青
    7: (0, 0, 0),        # 黑
    8: (153, 153, 153),  # 灰
}

UNOPENED = -1
EMPTY = 0
FLAG = -2          # 识别出旗子(回填后如需再读盘用)

TEMPLATE_SIZE = 28
DIGIT_CHROMA = 40

# 8 的字形特征: 两个孔; 0 只有一个孔. 灰色字形需用孔数区分 0 与 8.


def _glyph_hole_count(glyph):
    """数字形内部孔数(0 有 1 个孔, 8 有 2 个孔, 7 无孔)."""
    if glyph is None or glyph.size == 0 or glyph.max() == 0:
        return 0
    inv = cv2.bitwise_not(glyph)
    ncc, lab, stats, cent = cv2.connectedComponentsWithStats(inv, 8)
    holes = 0
    for i in range(1, ncc):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= inv.shape[1] or y + h >= inv.shape[0]:
            continue
        if area >= 3:
            holes += 1
    return holes


# ----------------------------------------------------------------------
# 单元格基本属性
# ----------------------------------------------------------------------

def cell_is_unopened(cell_bgr, cell_gray=None):
    """未翻开格特征: 全局高光或左上两条浮雕高光.

    阈值 0.06(非 0.04): 50×50 等小格子棋盘上, 数字格的高亮像素可到
    0.04-0.05, 但真正未翻开格的中位 bright 约 0.095, 阈值 0.06 能可靠区分.

    浏览器缩放会把横纵网格线分配到相邻像素, 个别未翻开格的整体高光比例
    会降到 0.06 以下。此时仍能看到非常稳定的 3D 浮雕特征: 上边和左边
    同时是亮边。数字格(包括灰色 0)没有这两条边, 因而用它作为缩放兜底.
    """
    if cell_gray is None:
        cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((cell_gray > 235).mean())
    if bright_ratio > 0.06:
        return True

    h, w = cell_gray.shape
    edge = max(2, int(round(min(h, w) * 0.12)))
    top_bright = float((cell_gray[:edge, :] > 235).mean())
    left_bright = float((cell_gray[:, :edge] > 235).mean())
    return top_bright > 0.18 and left_bright > 0.18




def _ink_mask(cell_bgr, cell_gray, margin_frac=0.12):
    """格子中心区域的墨迹掩码(彩色数字或深色数字)."""
    hh, ww = cell_gray.shape
    cy0, cy1 = int(hh * margin_frac), int(hh * (1 - margin_frac))
    cx0, cx1 = int(ww * margin_frac), int(ww * (1 - margin_frac))
    if cy1 - cy0 < 4 or cx1 - cx0 < 4:
        return None
    sub = cell_bgr[cy0:cy1, cx0:cx1]
    subg = cell_gray[cy0:cy1, cx0:cx1]
    chroma = (np.abs(sub[:, :, 0].astype(int) - sub[:, :, 1]) +
              np.abs(sub[:, :, 1] - sub[:, :, 2]) +
              np.abs(sub[:, :, 2] - sub[:, :, 0]))
    ink = (chroma > DIGIT_CHROMA) | ((subg < 170) & (chroma < DIGIT_CHROMA))
    if ink.sum() < 8:
        return None
    return ink


def classify_by_color(cell_bgr, cell_gray, chroma_floor=10):
    """颜色主分类: 返回 (数字, 中位墨色). 无墨迹时返回 (None, None).

    数字范围: 0-8. 0 表示"周围无雷"的数字, 与空白(无墨迹)不同.
    """
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is None:
        return None, None
    hh, ww = cell_gray.shape
    cy0, cy1 = int(hh * 0.12), int(hh * 0.88)
    cx0, cx1 = int(ww * 0.12), int(ww * 0.88)
    px = cell_bgr[cy0:cy1, cx0:cx1][ink]
    chp = (np.abs(px[:, 0].astype(int) - px[:, 1]) +
           np.abs(px[:, 1] - px[:, 2]) +
           np.abs(px[:, 2] - px[:, 0]))
    chrom_px = px[chp > DIGIT_CHROMA]
    if len(chrom_px) > chroma_floor:
        med = np.median(chrom_px, axis=0)
        best, bd = 0, 1e9
        for d, ref in COLOR_REF.items():
            dist = np.abs(med - np.array(ref)).sum()
            if dist < bd:
                bd, best = dist, d
        return best, med
    # 无彩色 -> 灰/黑数字: 灰=0(单环)或8(双环), 黑=7
    medg = np.median(px, axis=0)
    if medg.mean() > 70:
        # 灰色: 用孔数区分 0 与 8
        glyph = extract_glyph(cell_bgr, cell_gray)
        holes = _glyph_hole_count(glyph)
        return (8 if holes >= 2 else 0), medg
    return 7, medg


def extract_glyph(cell_bgr, cell_gray):
    """提取归一化 28x28 二值字形."""
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is None:
        return None
    hh, ww = cell_gray.shape
    cy0, cy1 = int(hh * 0.12), int(hh * 0.88)
    cx0, cx1 = int(ww * 0.12), int(ww * 0.88)
    g = (ink.astype(np.uint8)) * 255
    ys, xs = np.where(g)
    if len(ys) == 0:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = (TEMPLATE_SIZE - 6) / max(g.shape)
    nw = max(1, int(g.shape[1] * sc))
    nh = max(1, int(g.shape[0] * sc))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    y0c = (TEMPLATE_SIZE - nh) // 2
    x0c = (TEMPLATE_SIZE - nw) // 2
    canvas[y0c:y0c + nh, x0c:x0c + nw] = g
    return canvas


def classify_cell(cell_bgr):
    """综合分类一个格子: 返回 (值, 详情dict).

    值: -1 未翻开, 0 空白, 1-8 数字, -2 旗.
    注意: 数字 0 也是"数字"(周围无雷), 与"空白"不同.
    分类只依赖颜色/形状; 模板匹配对最终值无影响, 已移除(见 classify_screen).
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return EMPTY, {}
    cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 未翻开(高亮浮雕) -> 旗或未翻开
    if cell_is_unopened(cell_bgr, cell_gray):
        if _flag_like(cell_bgr):
            return FLAG, {"type": "flag"}
        return UNOPENED, {"type": "unopened"}

    # 2. 数字墨迹(彩色/深色)
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is not None:
        v_color, med = classify_by_color(cell_bgr, cell_gray)
        if v_color is not None:   # 有墨迹 => 是数字(含0), 不是空白
            return v_color, {"type": "digit", "color": v_color, "med": med}

    return EMPTY, {"type": "empty"}


def _flag_like(cell_bgr):
    """旗子特征: 红色三角旗面在格子上半部, 红色像素适中.

    与红色数字(3/5)区分: 数字居中(ycenter~0.47)且红色像素多(>80);
    旗子偏上(ycenter~0.35)且红色像素少(40-46).
    """
    b = cell_bgr[:, :, 0].astype(int)
    g = cell_bgr[:, :, 1].astype(int)
    r = cell_bgr[:, :, 2].astype(int)
    red = (r > 130) & (r > g + 60) & (r > b + 60)
    n_red = int(red.sum())
    if not (15 <= n_red <= 70):
        return False
    h, w = red.shape
    ys, xs = np.where(red)
    if len(ys) == 0:
        return False
    y_center = ys.mean() / h
    return y_center < 0.42


def classify_screen(img_bgr, grid, templates=None):
    """按规整网格把一屏切成格子并逐个分类.

    templates 参数保留仅为接口兼容(历史用途是模板匹配诊断); 分类结果完全
    由颜色/形状决定, 与模板无关。逐格对模板做 matchTemplate 会让大棋盘识别
    慢约 7 倍, 故不再使用。

    返回:
      values  m 行 x n 列 的整型数组(-1 未翻开 / 0 空白 / 1-8 数字)
      details 对应细节列表
      cells   BGR 格子列表(按行优先)
    """
    n = grid["n"]
    m = grid["m"]
    vx, hy = grid["vx"], grid["hy"]
    values = np.zeros((m, n), dtype=int)
    details = [[None] * n for _ in range(m)]
    cells = []
    for r in range(m):
        y0, y1 = hy[r], hy[r + 1]
        if y1 - y0 < 2:
            values[r, :] = UNOPENED
            continue
        for c in range(n):
            x0, x1 = vx[c], vx[c + 1]
            if x1 - x0 < 2:
                values[r, c] = UNOPENED
                continue
            # 内缩 2px 避开隔线
            cell = img_bgr[y0 + 2:y1 - 2, x0 + 2:x1 - 2]
            if cell.size == 0:
                values[r, c] = UNOPENED
                continue
            v, det = classify_cell(cell)
            values[r, c] = v
            details[r][c] = det
            cells.append(cell)
    return values, details, cells



# ======================================================================
# 四、滚动拼接识别与滚动回填(原 ms_auto.py)
# ======================================================================

def _print(msg):
    print(msg, flush=True)


def _screens_identical(a, b, grid, max_mean_diff=3.0):
    """判断两屏棋盘区域是否几乎完全一致(页面没有滚动).

    只用于识别“页面已到滚动极限/滚动无效”这一确定状态: 棋盘区域平均像素差
    必须极小。棋盘内容以未翻开格为主时, 滚动一整行后相邻行内容不同, 平均
    像素差通常大于 10(实测), 因此不会被误判为“没动”; 而真正没动时该差
    接近 0。网格线相位在整屏滚动时会回到原位, 不能作为“没动”的依据。
    """
    if a is None or b is None:
        return False
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        return False
    vx = grid.get("vx") or grid.get("vx_raw")
    hy = grid.get("hy") or grid.get("hy_raw")
    if not vx or not hy:
        return False
    x0, x1 = int(vx[0]), int(vx[-1])
    y0, y1 = int(hy[0]), int(hy[-1])
    x0, x1 = max(0, x0), min(ga.shape[1], x1)
    y0, y1 = max(0, y0), min(ga.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    diff = np.abs(ga[y0:y1, x0:x1].astype(int) - gb[y0:y1, x0:x1].astype(int))
    return float(diff.mean()) < max_mean_diff


class StitchedBoard:
    """拼接结果: 全局棋盘 + 每格来源信息(供回填)."""
    def __init__(self):
        self.n = 0
        self.pitch = 0.0
        self.board = None
        self.screens = []          # [(g0, hy_raw, vx, m)]
        self.screen_images = []    # 每屏截屏(BGR, 用于回填对齐)
        self.row_screen = {}       # 全局行号 -> screen_id
        self.x_screen = 0          # 截屏区域左边界物理像素
        self.y_screen = 0          # 截屏区域顶边物理像素
        self.bbox = None           # 实际使用的截屏区域(可能与用户点击略有差异)
        self.global_rows = {}      # 全局行号 -> 行数组
        self.global_detail = {}
        self.row_observations = {} # 全局行号 -> 多屏观察, 供冲突融合/诊断
        self.diagnostics = []      # 每屏定位与合并摘要
        self.reference_vx = None   # 棋盘横向不随滚动变化, 固定首屏列线


def _normalize_row(row):
    """用于跨屏比较的行: 旗子(-2)与未翻开(-1)视为同一状态."""
    out = np.asarray(row, dtype=int).copy()
    out[out < 0] = UNOPENED
    return out


def _row_signature(row):
    """位置敏感的稳定行指纹, 包含数字 0, 忽略未翻开/旗子.

    旧实现只保留数字值序列, 丢掉列号和数字 0, 不同行很容易得到相同指纹,
    从而把整屏映射到错误的全局行号.
    """
    norm = _normalize_row(row)
    sig = tuple((c, int(v)) for c, v in enumerate(norm) if v >= 0)
    return sig if sig else None


def _row_values_at(img_bgr, y_top, ref_vx, pitch):
    """读取 bbox 内 y_top 处一整行的格值(用参考列线切格).

    供回填点击前校验使用: 在拟点击的行位置重新读一行内容, 与题目原行比对,
    若数字大量冲突说明行号/坐标定位错位, 应停止而不是乱插旗。

    注意: classify_cell 对"无墨迹"返回 EMPTY=0, 与数字 0 同值。被视口边缘裁掉
    的未翻开格(浮雕高光被切)会读成 EMPTY; 这里把 type=="empty" 的格映射回
    UNOPENED, 避免把"被裁掉的未翻开"误当成已知数字 0 而产生假冲突。
    """
    n = len(ref_vx) - 1
    if n < 1:
        return None
    vals = np.zeros(n, dtype=int)
    y0 = int(round(y_top))
    for c in range(n):
        x0, x1 = int(ref_vx[c]), int(ref_vx[c + 1])
        cell = img_bgr[y0 + 2:y0 + int(pitch) - 2, x0 + 2:x1 - 2]
        if cell.size == 0:
            vals[c] = UNOPENED
            continue
        v, det = classify_cell(cell)
        if det is not None and det.get("type") == "empty":
            v = UNOPENED
        vals[c] = v
    return vals


def _fill_row_evidence(current, expected, expected_mines=(), completed=False):
    """比较回填阶段的一行，并区分稳定数字、灰化数字和旗帜证据。

    网页在插旗后会把一部分已经满足的彩色数字变成灰色。现有分类器会把灰色
    的 1/2/... 读成 0（灰色字形只专门区分了 0/8），所以回填定位不能再把
    ``当前=0/8, 原题正数字`` 当成硬冲突（灰字形的孔数会使结果落到 0 或 8）。
    该组合仍保留了“这一列有数字墨迹”这个位置证据，记为较弱的 gray_match。

    已完成区另有更强的状态证据：应为雷的位置若读到 FLAG，说明当前行号和列号
    同时对齐；非雷位置出现 FLAG 则是强冲突。未完成行不应出现任何旗帜。
    """
    cur = np.asarray(current, dtype=int)
    ref = np.asarray(expected, dtype=int)
    if cur.shape != ref.shape:
        return None

    n = cur.size
    expected_flag = np.zeros(n, dtype=bool)
    if completed:
        for col in expected_mines or ():
            if 0 <= int(col) < n:
                expected_flag[int(col)] = True

    flags = cur == FLAG
    flag_matches = int((flags & expected_flag).sum())
    flag_conflicts = int((flags & (~expected_flag)).sum())
    missing_flags = int(((~flags) & expected_flag).sum())

    cur_known = cur >= 0
    ref_known = ref >= 0
    exact_mask = cur_known & ref_known & (cur == ref)
    # 作答后灰化的正数字会被颜色分类器按孔数读成 0 或 8；保留为弱位置证据。
    gray_mask = cur_known & ((cur == 0) | (cur == 8)) & (ref > 0) & (cur != ref)
    conflict_mask = cur_known & (
        (~ref_known) | (ref_known & (cur != ref) & (~gray_mask)))
    missing_mask = (~cur_known) & ref_known

    exact = int(exact_mask.sum())
    gray_matches = int(gray_mask.sum())
    conflicts = int(conflict_mask.sum())
    missing = int(missing_mask.sum())
    score = (5.0 * exact + 2.0 * gray_matches
             + 9.0 * flag_matches
             - 12.0 * conflicts - 16.0 * flag_conflicts
             - 0.35 * missing - 0.10 * missing_flags)
    return {
        "score": score,
        "exact": exact,
        "gray_matches": gray_matches,
        "conflicts": conflicts,
        "missing": missing,
        "flag_matches": flag_matches,
        "flag_conflicts": flag_conflicts,
        "missing_flags": missing_flags,
    }


def _choose_bottomup_fill_window(values, global_rows, n, completed_from,
                                  mines_by_row):
    """给滚动后的整屏建立绝对行号映射（bottom-up 回填专用）。

    候选必须同时覆盖连续完成边界：屏内至少有一条尚未作答行，并且不能从
    ``completed_from`` 上方跳过去。打分组合两类互补证据：

    * 边界上方尚未作答行的数字/灰化数字位置；
    * 边界下方已完成行中，旗帜是否出现在本次求解得到的雷列。

    这样既不拿作答后的整行去硬匹配作答前整行，也不依赖滚轮一次移动多少
    像素。返回的 g0 直接对应 ``values[0]`` 和当前检测到的 ``hy[0]``。
    """
    if values is None:
        return None, {"reliable": False, "alternatives": []}
    vals = np.asarray(values, dtype=int)
    if (vals.ndim != 2 or vals.shape[0] < 1 or vals.shape[1] != n or
            completed_from <= 0):
        return None, {"reliable": False, "alternatives": []}

    visible = int(vals.shape[0])
    lo = max(0, int(completed_from) - visible)
    hi = min(n - 1, int(completed_from) - 1)
    scored = []
    for g0 in range(lo, hi + 1):
        # 连续边界必须仍在这一屏中；否则接受 g0 会静默跳过中间行。
        if g0 + visible < completed_from:
            continue
        total = {
            "score": 0.0, "exact": 0, "gray_matches": 0,
            "conflicts": 0, "missing": 0, "flag_matches": 0,
            "flag_conflicts": 0, "missing_flags": 0,
        }
        pending_rows = []
        for i, current in enumerate(vals):
            gr = g0 + i
            if gr < 0 or gr >= n or gr not in global_rows:
                continue
            completed = gr >= completed_from
            ev = _fill_row_evidence(
                current, global_rows[gr], mines_by_row.get(gr, ()), completed)
            if ev is None:
                continue
            for key in total:
                total[key] += ev[key]
            if not completed:
                pending_rows.append({"row": gr, **ev})

        pending_compatible = sum(
            row["exact"] + row["gray_matches"] for row in pending_rows)
        pending_conflicts = sum(
            row["conflicts"] + row["flag_conflicts"] for row in pending_rows)
        hard = total["exact"] + total["gray_matches"] + total["flag_matches"]
        summary = {
            "g0": g0, **total,
            "pending_rows": pending_rows,
            "pending_compatible": pending_compatible,
            "pending_conflicts": pending_conflicts,
            "hard_evidence": hard,
        }
        rank = (total["score"], total["flag_matches"],
                pending_compatible, total["exact"],
                -total["flag_conflicts"], -total["conflicts"], -g0)
        scored.append((rank, summary))

    if not scored:
        return None, {"reliable": False, "alternatives": []}
    scored.sort(key=lambda item: item[0], reverse=True)
    best = dict(scored[0][1])
    alternatives = [dict(item[1]) for item in scored[:3]]
    best["alternatives"] = alternatives
    runner_score = (float(scored[1][1]["score"])
                    if len(scored) > 1 else float("-inf"))
    margin = float(best["score"]) - runner_score
    best["score_margin"] = margin

    min_pending = max(3, min(8, n // 5))
    rows_individually_safe = bool(best["pending_rows"])
    for row in best["pending_rows"]:
        compatible = row["exact"] + row["gray_matches"]
        row_bad = row["conflicts"] + row["flag_conflicts"]
        known_expected = int((np.asarray(global_rows[row["row"]]) >= 0).sum())
        # 整行没有数字时本行自身没有内容锚点；允许由同屏其它未作答行和
        # 已完成区旗帜证明窗口位置，不能因此让连续边界永远跨不过空线索行。
        row_need = min(min_pending, known_expected) if known_expected > 0 else 0
        if compatible < row_need or row_bad > max(2, compatible // 4):
            rows_individually_safe = False
            break

    total_bad = best["conflicts"] + best["flag_conflicts"]
    pending_has_clues = any(
        int((np.asarray(global_rows[row["row"]]) >= 0).sum()) > 0
        for row in best["pending_rows"])
    pending_anchor_ok = best["pending_compatible"] >= min_pending
    if not pending_has_clues:
        # 极少数题会出现整条未翻开的行；这时只能由已处理区的正确旗列证明
        # 绝对行号，并仍要求候选有明显唯一优势。
        pending_anchor_ok = (
            best["flag_matches"] >= max(4, n // 4) and
            best["flag_conflicts"] == 0)
    reliable = (
        pending_anchor_ok and
        best["hard_evidence"] >= max(8, n // 3) and
        best["pending_conflicts"] <= max(2, best["pending_compatible"] // 4) and
        total_bad <= max(5, best["hard_evidence"] // 6) and
        margin >= 8.0 and rows_individually_safe)
    best["reliable"] = bool(reliable)
    return (int(best["g0"]) if reliable else None), best


def _verify_row_against_board(img_bgr, y_top, ref_vx, pitch, expected):
    """回填点击前校验: 未作答行的数字应与原题完全一致.

    从 y_top 读取一整行(约 pitch 像素)内容与 expected 对比。
    当行只露出大半时(顶边被视口裁掉), y_top 在顶部(≈0), 读到的内容缺失
    一部分, 部分格读成"空白"(UNOPENED), matched 数偏低但 flags 和冲突均低,
    此时判定为 "continue"(继续向上滚动)而不是 fatal。

    返回 (status, matched, conflicts, flags):
      status = "ok"      数字基本一致, 可安全插旗;
      status = "continue"行顶边被裁/邻行旗子渗入, 应继续向上滚动;
      status = "fatal"   数字冲突过多或整行都是旗, 行号/坐标错位, 停止回填。
    """
    cur = _row_values_at(img_bgr, y_top, ref_vx, pitch)
    if cur is None:
        return "fatal", 0, 0, 0
    evidence = _fill_row_evidence(cur, expected, (), completed=False)
    if evidence is None:
        return "fatal", 0, 0, 0
    # gray_matches 是“原题正数字在作答后灰化为 0”的兼容证据，不是冲突。
    matched = int(evidence["exact"] + evidence["gray_matches"])
    conflicts = int(evidence["conflicts"])
    flags = int((cur == FLAG).sum())
    # 一整行都是旗: 边界已越过已作答区(正在读已完成的行) -> 致命.
    if flags > 2:
        return "fatal", matched, conflicts, flags
    # 数字冲突过多且无旗: 读到的是其它行 -> 致命.
    if flags == 0 and conflicts > max(2, matched // 4):
        return "fatal", matched, conflicts, flags
    # 行在视口顶部(y_top 很小, 行被裁掉上半): 只对旗和明显错行 fatal,
    # 其余判定为 continue, 等该行完整进入视口后再插旗。
    if y_top < int(round(pitch * 0.4)):
        if flags > 0:
            return "continue", matched, conflicts, flags
        if conflicts > max(1, matched // 5):
            return "continue", matched, conflicts, flags
    # 有旗渗入但数字大体对: 该行尚未完整露出或刚露出 -> 继续滚动.
    if flags > 0:
        return "continue", matched, conflicts, flags
    expected_known = int((np.asarray(expected) >= 0).sum())
    need = min(3, expected_known)
    if matched < need:
        if y_top < int(round(pitch * 0.6)):
            return "continue", matched, conflicts, flags
        return "fatal", matched, conflicts, flags
    return "ok", matched, conflicts, flags


def _row_similarity(left, right):
    """给两次同行观察打分; 正数越大越像, 数字冲突会被重罚."""
    a = _normalize_row(left)
    b = _normalize_row(right)
    if a.shape != b.shape:
        return -1e9, 0, 0

    a_known = a >= 0
    b_known = b >= 0
    both_known = a_known & b_known
    same_known = both_known & (a == b)
    diff_known = both_known & (a != b)
    one_known = a_known ^ b_known
    both_hidden = (~a_known) & (~b_known)

    score = (4.0 * int(same_known.sum())
             - 7.0 * int(diff_known.sum())
             - 2.0 * int(one_known.sum())
             + 0.25 * int(both_hidden.sum()))
    if np.array_equal(a, b):
        score += 2.0 * a.size
    conflicts = int(diff_known.sum() + one_known.sum())
    agreements = int(same_known.sum() + both_hidden.sum())
    return score, agreements, conflicts


def _fuse_row_observations(observations):
    """融合同行的多屏观察, 优先保留信息更完整的数字行.

    真实故障表现为某次观察整行都是 -1。数字一旦在任一稳定截图中被看到，
    就不应再被另一张低质量的全 -1 行覆盖。若数字观察彼此冲突，则用出现
    次数、所在行的信息量和较早屏幕依次打破平局，并返回冲突数供日志提示.
    """
    if not observations:
        return None, None, None, 0

    def info_count(obs):
        return int((_normalize_row(obs["row"]) >= 0).sum())

    best = max(observations, key=lambda obs: (info_count(obs), -obs["sid"]))
    merged = _normalize_row(best["row"])
    chosen_detail = best["detail"]
    conflicts = 0

    for c in range(merged.shape[0]):
        votes = {}
        for obs in observations:
            v = int(_normalize_row(obs["row"])[c])
            if v < 0:
                continue
            # 信息更完整的行权重略高; 相同票数时仍保持确定性.
            weight = 1.0 + info_count(obs) / max(1.0, merged.shape[0])
            count, total_weight, first_sid = votes.get(v, (0, 0.0, obs["sid"]))
            votes[v] = (count + 1, total_weight + weight,
                        min(first_sid, obs["sid"]))
        if not votes:
            merged[c] = UNOPENED
            continue
        if len(votes) > 1:
            conflicts += 1
        winner = max(votes.items(),
                     key=lambda item: (item[1][0], item[1][1],
                                       -item[1][2], -item[0]))[0]
        merged[c] = winner

    return merged, chosen_detail, best["sid"], conflicts


def _choose_screen_g0(values, global_rows, predicted_g0, n,
                      previous_g0=0, search_radius=4):
    """用全屏重叠内容校正像素推算出的首行号.

    候选不仅来自像素预测附近, 也来自位置敏感行指纹的精确命中。最终由整屏
    所有重叠行共同投票, 避免旧实现用第一条碰巧重复的指纹决定整屏位置.
    """
    # 不能用 n-m 作为上限。真实页面滚到底部时, 网格检测可能把棋盘下方的
    # 深色横线当成额外分隔线, 使 m 比真实可见棋盘行数多 1~数行。此时正确
    # g0+m 可以超过 n, 超出棋盘的尾行由 add_rows 忽略即可。
    max_g0 = max(0, n - 1)
    predicted = int(np.clip(predicted_g0, 0, max_g0))
    min_allowed = int(np.clip(previous_g0, 0, max_g0))

    candidates = set(range(max(min_allowed, predicted - search_radius),
                           min(max_g0, predicted + search_radius) + 1))
    candidates.add(predicted)

    signature_index = {}
    for gr, row in global_rows.items():
        sig = _row_signature(row)
        if sig is not None:
            signature_index.setdefault(sig, []).append(gr)
    for i, row in enumerate(values):
        sig = _row_signature(row)
        if sig is None:
            continue
        for gr in signature_index.get(sig, []):
            candidate = gr - i
            if min_allowed <= candidate <= max_g0:
                candidates.add(candidate)

    best = None
    scored = []
    for candidate in sorted(candidates):
        score = 0.0
        overlaps = 0
        exact_rows = 0
        agreements = 0
        conflicts = 0
        for i, row in enumerate(values):
            gr = candidate + i
            if gr not in global_rows:
                continue
            overlaps += 1
            existing = global_rows[gr]
            row_score, row_agree, row_conflict = _row_similarity(existing, row)
            score += row_score
            agreements += row_agree
            conflicts += row_conflict
            if np.array_equal(_normalize_row(existing), _normalize_row(row)):
                exact_rows += 1

        # 内容分优先; 其余字段只负责稳定地打破平局.
        rank = (score, exact_rows, overlaps, agreements, -conflicts,
                -abs(candidate - predicted), -candidate)
        summary = {
            "g0": candidate, "score": score, "overlaps": overlaps,
            "exact_rows": exact_rows, "agreements": agreements,
            "conflicts": conflicts,
        }
        scored.append((rank, summary))
        if best is None or rank > best[0]:
            best = (rank, candidate, summary)

    if best is None:
        return predicted, {"score": 0.0, "overlaps": 0,
                           "exact_rows": 0, "agreements": 0,
                           "conflicts": 0}
    result = dict(best[2])
    scored.sort(key=lambda item: item[0], reverse=True)
    result["alternatives"] = [item[1] for item in scored[:3]]
    return best[1], result


def _screen_match_is_reliable(match, n):
    """拒绝会大面积污染已确认行的整屏错误定位/识别.

    小重叠(<5行)时, 必须至少有一行与已确认内容完全一致(数字或未翻开都算),
    否则说明位置没有可靠锚点(往往是滚过头、整屏错位), 不能接受。
    """
    overlaps = int(match.get("overlaps", 0))
    exact_rows = int(match.get("exact_rows", 0))
    conflicts = int(match.get("conflicts", 0))
    if overlaps < 5:
        return exact_rows >= 1
    exact_ratio = exact_rows / float(overlaps)
    conflict_ratio = conflicts / float(max(1, overlaps * n))
    return exact_ratio >= 0.50 and conflict_ratio <= 0.20


def _choose_fill_g0(values, global_rows, n):
    """插旗后的回填专用定位: 只用不会变化的数字格作为锚点.

    未翻开格插旗后外观会改变, 个别旗子还可能被分类成普通未翻开或红色墨迹,
    因而不能继续要求整行像识别阶段那样完全一致。数字 0..8 在作答前后不变,
    用其列位置和值对整屏候选 g0 打分更稳健.
    """
    scored = []
    for candidate in range(n):
        overlaps = 0
        matched = 0
        conflicts = 0
        missing = 0
        exact_digit_rows = 0
        for i, row in enumerate(values):
            gr = candidate + i
            if gr not in global_rows:
                continue
            overlaps += 1
            cur = _normalize_row(row)
            ref = _normalize_row(global_rows[gr])
            cur_known = cur >= 0
            ref_known = ref >= 0
            both = cur_known & ref_known
            matched += int((both & (cur == ref)).sum())
            conflicts += int((both & (cur != ref)).sum())
            # 当前把原本未翻开的格认成数字, 是比漏掉一个数字更强的错位信号.
            conflicts += int((cur_known & (~ref_known)).sum())
            missing += int(((~cur_known) & ref_known).sum())
            cur_sig = _row_signature(cur)
            if cur_sig is not None and cur_sig == _row_signature(ref):
                exact_digit_rows += 1

        score = (5.0 * matched - 12.0 * conflicts - 0.5 * missing
                 + 20.0 * exact_digit_rows)
        rank = (score, matched, exact_digit_rows, -conflicts, -missing,
                overlaps, -candidate)
        scored.append((rank, {
            "g0": candidate, "score": score, "overlaps": overlaps,
            "matched_digits": matched, "digit_conflicts": conflicts,
            "missing_digits": missing, "exact_digit_rows": exact_digit_rows,
        }))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = dict(scored[0][1])
    best["alternatives"] = [item[1] for item in scored[:3]]
    return int(best["g0"]), best


def _fill_match_is_reliable(match):
    matched = int(match.get("matched_digits", 0))
    conflicts = int(match.get("digit_conflicts", 0))
    missing = int(match.get("missing_digits", 0))
    total = matched + conflicts + missing
    if matched < 12 or total <= 0:
        return False
    return matched / float(total) >= 0.72 and conflicts <= max(3, int(matched * 0.08))


def _use_reference_columns(grid, reference_vx, n, pitch):
    """滚动只改变纵坐标; 始终复用首屏列线, 防止单屏纵线相位漂移."""
    stable = dict(grid)
    stable["n"] = n
    stable["pitch"] = pitch
    stable["vx"] = list(reference_vx)
    return stable


# ----------------------------------------------------------------------
# 滚动偏移(内容互相关 + 网格线精修)
# ----------------------------------------------------------------------

def compute_scroll_offset(prev_img, cur_img, prev_grid, cur_grid, prev_hy, cur_hy, pitch):
    """计算 cur 相对 prev 的滚动偏移(向下滚为正).

    以内容模板匹配为准; 若匹配分过低, 回退到网格线差值。

    旧实现回退时只改了 delta 却把内容匹配的低分原样返回, 导致调用方因
    “score < 0.6” 把网格线给出的可靠偏移误判为“无法对齐”。这里把网格线
    匹配到的线对数换算成与内容匹配一致的 0..1 置信度一并返回。
    """
    delta, score = align_screens(prev_img, cur_img, prev_grid, cur_grid, pitch)
    if score < 0.6:
        d2, s2 = match_scroll_delta(prev_hy, cur_hy, pitch)
        if s2 >= 3 and abs(d2) >= 1:
            # 线对数转 0..1: 3 条及以上共用网格线即视为高置信.
            line_conf = min(1.0, s2 / 3.0)
            if line_conf > score:
                delta, score = d2, line_conf
    return delta, score




# ----------------------------------------------------------------------
# 识别: 滚动拼接
# ----------------------------------------------------------------------

def refine_horizontal_bbox(bbox, stop, config, max_iter=6):
    """水平自适应微调: 把截屏区域向左右扩展到棋盘完整边界.

    用户点击的点只是大致范围, 若点到棋盘内侧(切开格子), 首/末列会被裁掉,
    导致列数推算偏小. 此函数逐步向外扩展截屏区域, 直到列数不再增加
    (即棋盘完整包含).

    返回 (refined_bbox, grid, img).
    """
    left, top, right, bottom = bbox
    scr_w, scr_h = get_screen_size()

    img, grid = None, None
    prev_n = None
    for _ in range(max_iter):
        img, grid = _grab_analyze_region((left, top, right, bottom))
        if grid is None:
            break
        cur_n = grid["n"]
        if prev_n is not None and cur_n <= prev_n:
            # 列数不再增加, 说明棋盘已完整
            break
        prev_n = cur_n
        pitch = grid["pitch"]
        vx = grid.get("vx_raw") or grid["vx"]
        if not vx:
            break
        margin_l = vx[0]
        margin_r = (right - left) - vx[-1]
        expand = 0
        if margin_l > 2:
            expand = max(expand, int(pitch * 1.1))
        if margin_r > 2:
            expand = max(expand, int(pitch * 1.1))
        if expand <= 0:
            break
        new_left = max(0, left - expand)
        new_right = min(scr_w, right + expand)
        if new_left == left and new_right == right:
            break
        left, right = new_left, new_right
    return (left, top, right, bottom), grid, img


def _grab_analyze_region(bbox):
    img = grab_screen(bbox)
    if img is None:
        return None, None
    grid = analyze_screen(img)
    return img, grid


def _board_line_thicknesses(img_bgr, hy_raw):
    """测每条已检测横线的暗带厚度(与 analyze_screen 的边框判据同一来源)."""
    gray = cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2GRAY)
    return [_band_thickness(gray, p, "h") for p in hy_raw]


def _left_border_top_y(gray, vx_raw, pitch):
    """左边框竖带内"持续暗带"的起点 = 棋盘顶边所在行.

    页面横线也会穿过左边框的 x 位置, 但只是几个像素高的短促暗段; 只有
    棋盘左边框会在该处形成贯穿整个棋盘高度的长暗带。取"该行本身为暗,
    且其后一个格距窗口内暗行占比过半"的第一行, 即棋盘顶边——对细线
    (缩放后边框不比隔线粗)、断续/虚线边框均稳健; 页面横线因窗口占比低
    被自然跳过。
    """
    if not vx_raw:
        return None
    h, w = gray.shape
    x = int(vx_raw[0])
    strip = gray[:, max(0, x - 1):min(w, x + 2)]
    if strip.size == 0 or h < 10:
        return None
    dark = ((strip < DARK_GRAY).any(axis=1)).astype(np.float64)
    win = max(8, int(round(pitch)))
    if h <= win:
        return None
    csum = np.concatenate([[0.0], np.cumsum(dark)])
    for y in range(0, h - win):
        # 该行是暗带起点(本身为暗), 且其后一个格距内持续为暗(占比过半)
        if dark[y] and csum[y + win] - csum[y] >= 0.6 * win:
            return y
    return None


def _anchor_board_top(bbox, grid, img):
    """把识别起点锚定到棋盘真实的加粗顶边框.

    框选(或点选)的顶边高于棋盘顶边框时, 顶边框之上会混入页面杂线或被
    裁掉一半的未见行; 它们与边框的间隔会被 count_grid_intervals 算成
    多余的"第 0 行", 使棋盘整体落到全局行 1..n, 此后 confirm_real_bottom
    要求的 "g0 + local_rows == n" 永远无法满足, 程序会不停下滚直到无进展
    报错退出。这里在已截画面内定位加粗顶边框并裁掉其上方内容:

      - 最佳加粗候选就是首条横线 -> 原样返回(点选贴边框的正常情形, 零改动)
      - 顶边框在画面中部 -> 多候选时用"左边框长暗带起点=棋盘顶边"定位
        真边框(可识别恰好压在边框上方一格的同厚页面粗线), 纯内存裁剪后
        用 analyze_screen 复检首线确为顶边框
      - 画面内没有顶边框候选且顶边上方还有屏幕空间 -> 向上扩大截屏范围
        再找(页面静止, 不滚动; 框选完整题目时顶边框必在屏幕内)
      - 无法锚定时原样返回, 由下游"确认真实底边"安全门兜底
    """
    if grid is None or img is None or img.size == 0:
        return bbox, grid, img

    left, top, right, bottom = bbox
    pitch = float(grid.get("pitch") or 0.0)
    if pitch <= 0:
        return bbox, grid, img

    cur_bbox, cur_grid, cur_img = bbox, grid, img
    for _ in range(5):
        hy = [int(p) for p in (cur_grid.get("hy_raw") or [])]
        if len(hy) < MIN_LINES:
            break
        thickness = _board_line_thicknesses(cur_img, hy)
        interior = thickness[1:-1] or thickness
        border_thr = max(3.0, float(np.median(interior)) + 1.5)
        lo, hi = 0.8 * pitch, 1.25 * pitch
        n_guess = int(cur_grid.get("n") or 0)

        def run_below(idx):
            """该线下方连续约一格间距的行数; 真顶边框此值最长."""
            run = 0
            for a, b in zip(hy[idx:], hy[idx + 1:]):
                if lo <= (b - a) <= hi:
                    run += 1
                else:
                    break
            return run

        # 左边框长暗带起点 = 棋盘顶边(不依赖线条厚度, 细线/虚线均稳健)
        gray = cv2.cvtColor(np.ascontiguousarray(cur_img), cv2.COLOR_BGR2GRAY)
        top_y = _left_border_top_y(gray, cur_grid.get("vx_raw") or [], pitch)

        # 底边框下方没有行(run=0), 天然排除
        cands = [i for i, t in enumerate(thickness)
                 if t >= border_thr and run_below(i) >= 1]
        if cands:
            # 多个加粗候选时, 优先取位置与棋盘顶边吻合者, 再按运行长度
            if top_y is not None:
                ranked = sorted(
                    cands,
                    key=lambda i: (abs(hy[i] - top_y) <= 8, run_below(i), i),
                    reverse=True)
            else:
                ranked = sorted(cands, key=lambda i: (run_below(i), i),
                                reverse=True)
            if ranked[0] == 0 and (top_y is None or abs(hy[0] - top_y) <= 8):
                return cur_bbox, cur_grid, cur_img     # 首线即顶边框, 无需裁剪
            anchored = None
            for ci in ranked:
                by, bt = hy[ci], thickness[ci]
                crop = max(0, by - (int(bt) // 2 + 3))
                sub = np.ascontiguousarray(cur_img[crop:, :])
                g2 = analyze_screen(sub)
                # 复检: 首线确为加粗边框, 且其位置与棋盘顶边吻合
                # (防止裁到恰好压在边框上方一格的同厚页面粗线)
                if (g2 is not None and g2.get("top_is_border")
                        and g2.get("n") == cur_grid.get("n")
                        and (top_y is None
                             or abs(float(g2["hy_raw"][0]) - (top_y - crop))
                             <= 8)):
                    anchored = ((left, cur_bbox[1] + crop, right, bottom), g2, sub)
                    break
            if anchored is not None:
                return anchored
            return bbox, grid, img      # 候选复检失败: 放弃锚定, 保持原样
        # 无加粗线(部分缩放下边框不比隔线粗): 用棋盘顶边位置做几何锚定
        if top_y is not None and n_guess >= 2:
            geo = [i for i, yv in enumerate(hy)
                   if abs(yv - top_y) <= 8 and run_below(i) >= n_guess - 2]
            for gi in geo:
                if gi == 0:
                    return cur_bbox, cur_grid, cur_img   # 首线即顶边
                crop = max(0, hy[gi] - 6)
                sub = np.ascontiguousarray(cur_img[crop:, :])
                g2 = analyze_screen(sub)
                g2_hy = (g2 or {}).get("hy_raw") or []
                if (g2 is not None and g2.get("n") == cur_grid.get("n")
                        and len(g2_hy) >= 2
                        and g2_hy[0] <= max(12.0, pitch * 0.5)
                        and abs(g2_hy[0] - (top_y - crop)) <= 8
                        and 0.6 * pitch <= g2_hy[1] - g2_hy[0] <= 1.4 * pitch):
                    return ((left, cur_bbox[1] + crop, right, bottom), g2, sub)
            if geo:
                return bbox, grid, img      # 几何候选复检失败: 保持原样
        # 画面内没有顶边框: 上方有屏幕空间则扩大截屏范围再找(不滚动页面)
        cur_top = cur_bbox[1]
        if cur_top <= 0:
            break
        new_top = max(0, cur_top - int(max(pitch * 3.0, 120.0)))
        img2 = grab_screen((left, new_top, right, bottom))
        g2 = analyze_screen(img2) if img2 is not None else None
        if g2 is None:
            break
        cur_bbox, cur_grid, cur_img = (left, new_top, right, bottom), g2, img2
    return bbox, grid, img


def _locate_board(bbox, stop, config):
    """定位棋盘, 返回 (bbox, grid, img).

    策略(锚定真实顶边框): 用户给出的顶边只决定初始截屏范围, 识别起点是
    检测到的棋盘加粗顶边框.
      - 先水平微调(确保棋盘完整横向)
      - 首条横线已是加粗顶边框 -> 直接以当前画面开始(g0=0 = 顶边框),
        与旧版点选贴边框的习惯一致
      - 顶边框之上混入页面杂线/半行(框选带冗余边距时) -> 纯内存裁剪,
        以顶边框为识别起点, 避免多余的"第 0 行"挤占全局行号
      - 画面内没有顶边框且上方有屏幕空间 -> 向上扩大截屏范围再找(不滚动)
      - 完全检测不到棋盘网格时, 保持原有向上扩展 / 下滚找棋盘的逻辑
    """
    left, top, right, bottom = bbox
    scr_w, scr_h = get_screen_size()
    img, grid = None, None

    # 先水平微调(确保棋盘完整横向)
    bbox, grid, img = refine_horizontal_bbox((left, top, right, bottom), stop, config)
    left, top, right, bottom = bbox

    if grid is None:
        # 检测不到网格: 向上扩展 bbox 再检测
        for _ in range(10):
            img, grid = _grab_analyze_region((left, top, right, bottom))
            if grid is not None:
                break
            new_top = max(0, top - int((right - left) * 0.7))
            if new_top == top:
                break
            top = new_top

        # 仍找不到: 向下滚动找
        if grid is None:
            img, grid = _grab_analyze_region((left, top, right, bottom))
            if grid is None:
                for _ in range(20):
                    scroll_wheel(config.get("scroll_amount", -60))
                    time.sleep(config.get("scroll_settle", 0.2))
                    img, grid = _grab_analyze_region((left, top, right, bottom))
                    if grid is not None:
                        break
        bbox = (left, top, right, bottom)

    # 锚定真实顶边框(框选允许带冗余边距; 点选贴边框时零改动)
    return _anchor_board_top(bbox, grid, img)


def stitch_recognize(bbox, stop, config, on_screen=None):
    """滚动拼接识别. 返回 StitchedBoard.

    定位核心: 累计像素偏移先给出行号预测, 再由位置敏感行指纹和整屏重叠
    内容共同校正。每个全局行保留多屏观察并融合, 避免一次全 -1 误识别
    永久覆盖正确内容.
    """
    left, top, right, bottom = bbox
    st = StitchedBoard()
    st.x_screen = left
    st.y_screen = top

    def grab_and_analyze():
        img = grab_screen(bbox)
        if img is None:
            return None, None
        grid = analyze_screen(img)
        return img, grid

    def classify(grid, img):
        values, details, _ = classify_screen(img, grid, templates=config.get("templates"))
        return values, details

    # ---------- 0/1. 定位棋盘(水平微调 + 尊重用户点选) ----------
    bbox, grid, img = _locate_board(bbox, stop, config)
    left, top, right, bottom = bbox
    st.x_screen = left
    st.y_screen = top
    st.bbox = bbox

    if grid is None:
        _print("[错误] 无法检测到棋盘网格, 请确认框选区域覆盖棋盘")
        return None

    if on_screen:
        on_screen(0, img.copy())

    n = grid["n"]
    st.n = n
    st.pitch = grid["pitch"]
    st.reference_vx = list(grid["vx"])
    _print(f"[识别] 列数 n={n}, 间距 {st.pitch:.1f}px")

    g0 = 0
    values, details = classify(grid, img)
    global_rows = {}
    global_detail = {}
    row_observations = {}

    def add_rows(vals, dets, sid, g0_cur):
        """按合法全局行号收集观察并融合; 已有行允许被更好观察修正."""
        n_rows, _ = vals.shape
        added = 0
        improved = 0
        conflicts = 0
        ignored = 0
        for i in range(n_rows):
            gr = g0_cur + i
            if gr < 0 or gr >= n:
                ignored += 1
                continue
            existed = gr in global_rows
            before = global_rows.get(gr)
            row_observations.setdefault(gr, []).append({
                "row": vals[i].copy(),
                "detail": dets[i],
                "sid": sid,
            })
            merged, merged_detail, source_sid, row_conflicts = \
                _fuse_row_observations(row_observations[gr])
            global_rows[gr] = merged
            global_detail[gr] = merged_detail
            st.row_screen[gr] = source_sid
            conflicts += row_conflicts
            if not existed:
                added += 1
            elif not np.array_equal(before, merged):
                improved += 1
        return added, improved, conflicts, ignored

    def confirm_real_bottom(screen_grid, g0_cur):
        """确认当前屏确实看到了棋盘第 n 行和真实底边。

        仅仅收集到 n 个行号不够：视口底沿的深色线可能制造一条伪行。
        网格检测必须在截图内部找到粗底边，底边前一格必须具有完整行高，
        且该底边按当前 g0 正好对应全局第 n 条边界。本项目题目保证唯一
        可解；实机题的完整末行有数字，因此整行仍为 -1 说明只截到了局部
        行，必须继续滚动，绝不能放行求解。
        """
        candidates = list(screen_grid.get("bottom_border_rows", []))
        matching = [local_rows for local_rows in candidates
                    if g0_cur + local_rows == n]
        last_known = 0
        if n - 1 in global_rows:
            last_known = int((_normalize_row(global_rows[n - 1]) >= 0).sum())
        confirmed = bool(matching) and last_known > 0
        return confirmed, {
            "candidates": candidates,
            "matching": matching,
            "last_known": last_known,
            "details": list(screen_grid.get("bottom_border_details", [])),
            "rejected": list(screen_grid.get("rejected_bottom_borders", [])),
        }

    added, improved, conflicts, ignored = add_rows(values, details, 0, g0)
    bottom_confirmed, bottom_info = confirm_real_bottom(grid, g0)
    st.screens.append((g0, grid["hy_raw"], grid["vx"], values.shape[0]))
    st.screen_images.append(img.copy())
    st.diagnostics.append({
        "sid": 0, "g0": 0, "g0_pred": 0, "delta": 0.0,
        "cumulative_scroll": 0.0, "added": added, "improved": improved,
        "conflicts": conflicts, "ignored": ignored,
    })
    board_complete = len(global_rows) >= n and bottom_confirmed
    if board_complete:
        _print(f"[识别] 棋盘完整可见且已确认底边, "
               f"末行有效数字={bottom_info['last_known']}, 无需滚动")
    else:
        _print("[识别] 棋盘超出屏幕, 开始滚动拼接...")

    prev_img = img
    prev_hy = grid["hy_raw"]
    prev_grid = grid
    prev_g0 = g0
    first_hy = float(grid["hy"][0])
    cumulative_scroll = 0.0
    screen_id = 1
    zero_add_streak = 0
    low_quality_streak = 0
    max_no_progress = config.get("max_no_progress", 3)
    base_scroll = config.get("scroll_amount", -60)
    max_attempts = config.get("scroll_retry_attempts", 6)
    current_scroll = base_scroll          # 自适应滚动量, 避免每次重置成大跳

    # ---------- 2. 滚动拼接 ----------
    while not bottom_confirmed and not stop.stopped:
        img, gcur = None, None
        attempt = 0
        scroll_now = current_scroll
        aligned_ok = False
        no_movement = False
        best = None
        while attempt < max_attempts:
            scroll_wheel(scroll_now)
            time.sleep(config.get("scroll_settle", 0.25))
            img, gcur = grab_and_analyze()
            if on_screen and img is not None:
                on_screen(screen_id, img.copy())
            if gcur is None:
                attempt += 1
                scroll_now = int(scroll_now * 0.5)
                continue
            cur_hy = gcur["hy_raw"]
            # 页面是否真的没动: 只有内容像素级一致才算(棋盘区域平均差极小)。
            # 不能用“对齐位移≈0”来判断——当滚动恰好一整屏(网格线相位回到原位)
            # 时, 网格线差值会给 delta≈0, 但内容其实已经整屏换行。
            if _screens_identical(prev_img, img, gcur):
                no_movement = True
                best = (0.0, 1.0, img, gcur)
                break
            delta, score = compute_scroll_offset(prev_img, img, prev_grid, gcur,
                                                 prev_hy, cur_hy, st.pitch)
            if abs(delta) >= 1 and score >= 0.6:
                aligned_ok = True
                best = (delta, score, img, gcur)
                break
            # 未对齐: 滚回去并用更小的滚动量重试(避免一次滚过头丢重叠).
            scroll_wheel(-scroll_now)
            time.sleep(config.get("scroll_settle", 0.2))
            scroll_now = int(scroll_now * 0.6)
            attempt += 1

        if not aligned_ok and not no_movement:
            zero_add_streak += 1
            if zero_add_streak >= max_no_progress:
                break
            # 整轮失败, 下一次从更保守的滚动量开始, 避免再次大跳.
            if abs(current_scroll) > 10:
                current_scroll = int(current_scroll * 0.6)
            continue

        delta, score, img, gcur = best

        if no_movement:
            # 到滚动极限: 当前画面与上一屏相同, g0 不变. 在当前位置确认底边;
            # 无论是否确认都停止, 绝不上下循环.
            bottom_confirmed, bottom_info = confirm_real_bottom(gcur, prev_g0)
            st.screens.append((prev_g0, gcur["hy_raw"],
                               gcur.get("vx") or st.reference_vx,
                               int(gcur.get("m", 0))))
            st.screen_images.append(img.copy())
            st.diagnostics.append({
                "sid": screen_id, "no_movement": True, "g0": prev_g0,
                "g0_pred": prev_g0, "delta": 0.0, "align_score": 1.0,
                "cumulative_scroll": cumulative_scroll, "added": 0,
                "improved": 0, "conflicts": 0, "ignored": 0,
                "visible_rows": int(gcur.get("m", 0)), "x_drift": 0.0,
                "bottom_confirmed": bottom_confirmed,
                "bottom_info": bottom_info,
            })
            break

        detected_vx = np.asarray(gcur.get("vx", []), dtype=float)
        reference_vx = np.asarray(st.reference_vx, dtype=float)
        if detected_vx.shape == reference_vx.shape and detected_vx.size:
            x_drift = float(np.max(np.abs(detected_vx - reference_vx)))
        else:
            x_drift = float("inf")
        stable_grid = _use_reference_columns(gcur, st.reference_vx, n, st.pitch)
        values, details = classify(stable_grid, img)
        candidate_scroll = cumulative_scroll + float(delta)
        # 不对每次 delta/pitch 单独四舍五入, 避免小误差逐屏累积。首条网格线
        # 的相位变化也必须计入, 才能得到当前第一条可见线的真实全局行号.
        g0_pred = int(round((candidate_scroll + float(gcur["hy"][0])
                             - first_hy) / st.pitch))
        g0_cur, match = _choose_screen_g0(
            values, global_rows, g0_pred, n, previous_g0=prev_g0,
            search_radius=config.get("g0_search_radius", 4))

        # 与已确认棋盘几乎没有可靠重叠 => 本次滚动跳过了重叠区, 位置不可信。
        # 旧实现会接受这种“无证据”屏, 用像素预测猜 g0, 导致行号整体偏移。
        # 这里回滚并缩小滚动量重试, 强制后续屏与已确认内容可靠重叠后再拼接。
        overlaps = int(match.get("overlaps", 0))
        exact_rows = int(match.get("exact_rows", 0))
        if len(global_rows) > 0 and (overlaps < 3 and exact_rows < 2):
            if abs(scroll_now) >= 1:
                scroll_wheel(-scroll_now)
                time.sleep(config.get("scroll_settle", 0.2))
            zero_add_streak += 1
            if abs(current_scroll) > 10:
                current_scroll = int(current_scroll * 0.6)
            st.diagnostics.append({
                "sid": screen_id, "no_overlap": True, "g0": g0_cur,
                "g0_pred": g0_pred, "delta": float(delta),
                "visible_rows": int(values.shape[0]),
                "match": match,
            })
            if zero_add_streak >= max_no_progress:
                break
            continue

        if not _screen_match_is_reliable(match, n):
            low_quality_streak += 1
            _print(f"[识别] 已安全丢弃低质量滚动画面 {screen_id}")
            st.diagnostics.append({
                "sid": screen_id, "rejected": True, "g0": g0_cur,
                "g0_pred": g0_pred, "delta": float(delta),
                "candidate_scroll": candidate_scroll,
                "visible_rows": int(values.shape[0]),
                "x_drift": x_drift, "match": match,
            })
            if low_quality_streak >= config.get("max_bad_screens", 3):
                _print("[错误] 连续多屏无法与已确认棋盘可靠对齐, "
                       "为避免错误作答, 本次识别终止")
                return None
            time.sleep(config.get("scroll_settle", 0.25))
            continue

        low_quality_streak = 0
        # 用最终融合的 g0 反推累计滚动, 而不是简单累加 delta。当内容匹配把
        # g0 固定在某处时, 若继续无限累加 delta, cumulative_scroll 会漂移到
        # 远超棋盘范围, 使像素预测(g0_pred)彻底失真。
        cumulative_scroll = (float(g0_cur) * st.pitch
                             - (float(gcur["hy"][0]) - first_hy))

        added, improved, conflicts, ignored = add_rows(
            values, details, screen_id, g0_cur)
        bottom_confirmed, bottom_info = confirm_real_bottom(gcur, g0_cur)
        st.screens.append((g0_cur, cur_hy, stable_grid["vx"], values.shape[0]))
        st.screen_images.append(img.copy())
        st.diagnostics.append({
            "sid": screen_id, "g0": g0_cur, "g0_pred": g0_pred,
            "delta": float(delta), "align_score": float(score),
            "cumulative_scroll": cumulative_scroll, "added": added,
            "improved": improved, "conflicts": conflicts,
            "ignored": ignored, "visible_rows": int(values.shape[0]),
            "x_drift": x_drift, "match": match,
            "bottom_confirmed": bottom_confirmed,
            "bottom_info": bottom_info,
        })

        if bottom_confirmed:
            _print(f"[识别] 已确认真实底边，完整收录 {len(global_rows)}/{n} 行")

        if added == 0 and not bottom_confirmed:
            zero_add_streak += 1
            if zero_add_streak >= max_no_progress:
                break
        else:
            zero_add_streak = 0

        # 自适应滚动量: 根据这次实际移动像素与可见行高调整下一次, 保证前后屏
        # 始终有重叠, 避免一次滚过头丢内容。只向下滚动时调整, 向上不调整.
        if delta > 0:
            visible_h = float(values.shape[0]) * st.pitch
            target_delta = visible_h * 0.55
            if delta > target_delta and abs(scroll_now) >= 2:
                current_scroll = -max(2, int(abs(scroll_now) * target_delta / delta))
            else:
                current_scroll = scroll_now
        else:
            current_scroll = scroll_now

        prev_img = img
        prev_hy = cur_hy
        prev_grid = gcur
        prev_g0 = g0_cur
        screen_id += 1

    # ---------- 3. 组装全局棋盘 ----------
    if not bottom_confirmed:
        _print("[错误] 未能在截图内部确认棋盘真实底边; "
               "为避免漏看题目，本次识别已终止，不进入求解")
        return None

    missing = [r for r in range(n) if r not in global_rows]
    if missing:
        human_rows = [r + 1 for r in missing]
        _print(f"[错误] 拼接仍缺失 {len(missing)} 行(按 1 起始): {human_rows}. "
               "为避免把缺行伪装成未翻开并错误提交, 本次识别已终止")
        return None

    board = np.zeros((n, n), dtype=int)
    for r in range(n):
        row = global_rows[r]
        if row.shape[0] != n:
            if row.shape[0] > n:
                row = row[:n]
            else:
                row = np.concatenate([row, np.full(n - row.shape[0], -1, dtype=int)])
        board[r] = row

    st.board = board
    st.global_rows = global_rows
    st.global_detail = global_detail
    st.row_observations = row_observations
    return st


# ----------------------------------------------------------------------
# 框选模式: 单屏识别(题目完整可见, 全程不滚动)
# ----------------------------------------------------------------------

def _locate_board_static(bbox, stop, config):
    """框选模式定位: 只做水平微调 + 锚定顶边框, 不滚动、不向下找棋盘.

    与 _locate_board 的区别: 框选完整题目时网格必然就在框内, 检测不到
    只能说明框错了, 应安全失败而不是去别处滚动找。
    """
    bbox, grid, img = refine_horizontal_bbox(bbox, stop, config)
    return _anchor_board_top(bbox, grid, img)


def recognize_visible_board(bbox, stop, config, on_screen=None):
    """框选模式识别: 题目完整可见于框内, 单屏识别, 全程不滚动.

    返回与滚动拼接同构的 StitchedBoard, 求解/回填/校验直接复用。
    完整性对齐其他项目: 框 = 完整题目, 底由框定义, 框内检测到 n 个
    完整行即完整(不做底边框判定)。行数不足时安全返回 None, 由调用方
    提示重新框选或改用点击两点模式(超大棋盘)。
    """
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        _print("[错误] 框选区域无效")
        return None
    img = grab_screen(bbox)
    if img is None:
        _print("[错误] 框选区域截屏失败")
        return None
    if analyze_screen(img) is None:
        _print("[错误] 框内未检测到棋盘网格, 请重新框选完整题目")
        return None

    # 水平微调 + 锚定棋盘加粗顶边框(纯内存裁剪/静态扩查, 不滚动)
    bbox, grid, img = _locate_board_static(bbox, stop, config)
    if grid is None:
        _print("[错误] 框内未检测到棋盘网格, 请重新框选完整题目")
        return None
    left, top, right, bottom = bbox
    if on_screen:
        on_screen(0, img.copy())

    n = grid["n"]
    st = StitchedBoard()
    st.n = n
    st.pitch = grid["pitch"]
    st.reference_vx = list(grid["vx"])
    st.x_screen = left
    st.y_screen = top
    st.bbox = bbox

    values, details, _ = classify_screen(
        img, grid, templates=config.get("templates"))
    m = int(values.shape[0])

    # 完整性(对齐高楼/帐篷/数墙/数独的框选语义): 框 = 完整题目, 底由
    # 框定义 —— 框内检测到 n 个完整行即完整。不做任何底边框判定(厚度/
    # 几何), 那是滚动模式判断"是否滚到题底"的判据, 与框选无关; 框内
    # 棋盘下方的页面伪行照旧忽略(只取前 n 行)。
    if m < n:
        _print(f"[错误] 框内只看到 {m}/{n} 个完整行, 题目未框全。"
               "请重新框选完整题目(可略大于棋盘); "
               "若题目超出屏幕请改用点击两点模式")
        return None

    board = np.zeros((n, n), dtype=int)
    for r in range(n):
        row = values[r]
        if row.shape[0] > n:
            row = row[:n]
        elif row.shape[0] < n:
            row = np.concatenate(
                [row, np.full(n - row.shape[0], -1, dtype=int)])
        board[r] = row

    # 截断保护(内容判定, 与边框无关): 末行整行未识别到数字而上一行有
    # 数字, 说明末行被框沿裁掉(裁边残条被当成了一行), 须重新框选。
    if (n >= 2 and (board[n - 1] == UNOPENED).all()
            and int((board[n - 2] >= 0).sum()) > 0):
        _print("[错误] 末行整行没有识别到数字而上一行有数字: "
               "末行大概率被框选下沿裁掉。请把框的下沿下移后重新框选")
        return None

    st.board = board
    st.global_rows = {r: board[r] for r in range(n)}
    st.global_detail = {r: details[r] for r in range(n)}
    st.row_observations = {
        r: [{"row": values[r].copy(), "detail": details[r], "sid": 0}]
        for r in range(n)}
    st.row_screen = {r: 0 for r in range(n)}
    st.screens = [(0, grid["hy_raw"], grid["vx"], m)]
    st.screen_images = [img.copy()]
    st.diagnostics = [{
        "sid": 0, "mode": "box", "g0": 0, "visible_rows": m,
        "bottom_defined_by_frame": True,
    }]
    _print(f"[识别] 框选单屏完成: {n}x{n}, 可见 {m} 行, 未翻开 "
           f"{int((board == UNOPENED).sum())} 格, 全程未滚动")
    return st


# ----------------------------------------------------------------------
# 回填: 滚动标记雷
# ----------------------------------------------------------------------

def fill_from_board(st, bbox, stop, config, mine_cells, debug_cb=None):
    """滚动回填: 右键标记所有雷.

    bottomup 模式只在初始底屏做旧题整屏定位。之后维护连续完成行边界，
    每次滚动都用未作答数字和已插旗状态重建绝对行号，再由当前网格换算坐标。
    topdown 兼容模式仍使用数字锚点逐屏定位。

    返回成功标记数.
    """
    left, top, right, bottom = bbox
    n = st.n
    pitch = st.pitch
    direction = config.get("fill_direction", "bottomup")

    # 雷按行分组；整行批次即使没有雷也会推进完成边界。
    mines_by_row = {}
    for r, c in mine_cells:
        mines_by_row.setdefault(r, []).append(c)

    target_rows = sorted(mines_by_row.keys(), reverse=(direction == "bottomup"))
    if not target_rows:
        _print("[回填] 没有需要标记的雷")
        return 0

    def locate_current(img, grid, retries=3):
        """定位插旗后的当前屏; 失败时在原位置重截, 不额外滚动."""
        last_vals = None
        last_grid = grid
        for attempt in range(retries):
            if img is None or grid is None:
                img, grid = _grab_analyze(bbox)
            if img is not None and grid is not None:
                stable_grid = _use_reference_columns(
                    grid, st.reference_vx or grid["vx"], n, pitch)
                vals, _, _ = classify_screen(
                    img, stable_grid, templates=config.get("templates"))
                g0_found, match = _choose_fill_g0(vals, st.global_rows, n)
                if _fill_match_is_reliable(match):
                    return g0_found, vals, stable_grid, img
                last_vals = vals
                last_grid = stable_grid
            if attempt + 1 < retries:
                time.sleep(config.get("fill_recheck_settle", 0.35))
                img, grid = _grab_analyze(bbox)
        return None, last_vals, last_grid, img

    # 初始画面停在识别阶段的底部, 只定位一次；同一屏内连续标记时网格不变.
    img, gcur = _grab_analyze(bbox)
    g0_cur, vals, gcur, img = locate_current(img, gcur)
    if g0_cur is None:
        _print("[错误] 回填初始截屏无法可靠定位")
        return 0

    filled = 0

    # 自底向上回填时，不再在每次滚动后拿“已经插满旗的整屏”与原题重做
    # 全局匹配。维护 completed_from：从该行到棋盘底部都已经按整行处理
    # 完毕。滚动后用未作答区域与旗帜状态重建绝对行号；新行必须紧接在
    # completed_from 之前。这样滚动步长不固定、旗子增多也不会污染定位。
    if direction == "bottomup":
        completed_from = n
        max_scrolls = int(config.get("max_fill_scroll_steps", 50))
        scroll_count = 0

        def focus_current_board(screen_grid):
            """每次滚轮前把鼠标重新放到当前可见棋盘的安全中心。"""
            vx = screen_grid.get("vx") or []
            hy = screen_grid.get("hy") or []
            if len(vx) < 2 or len(hy) < 2:
                return False
            x = (float(vx[0]) + float(vx[-1])) / 2.0
            y = (float(hy[0]) + float(hy[-1])) / 2.0
            # 留在 bbox 内部，避免最后一条线恰好贴着任务栏/视口边缘。
            x = min(max(x, 2.0), max(2.0, float(right - left) - 3.0))
            y = min(max(y, 2.0), max(2.0, float(bottom - top) - 3.0))
            move_mouse(left + x, top + y)
            return True

        def click_new_rows(first_row, old_boundary, screen_grid, verify=None):
            """把 [first_row, old_boundary) 的整行全部处理完。

            verify(row, y_top) 若提供, 在点击前先整体校验本批所有行:
              status "ok"       -> 继续, 全部通过后统一插旗;
              status "continue" -> 某行尚未完整露出(顶边被裁), 本批一个都不点,
                                   返回 "continue" 让外层继续向上滚动;
              status "fatal"    -> 行号/坐标错位, 停止回填。
            返回 "ok" / "continue" / "fatal"。
            """
            nonlocal filled
            hy = screen_grid.get("hy") or []
            vx = screen_grid.get("vx") or []
            visible_rows = max(0, len(hy) - 1)
            if (first_row < 0 or first_row >= old_boundary or
                    old_boundary > first_row + visible_rows):
                _print(f"[错误] 新行范围 {first_row + 1}-{old_boundary} "
                       "没有完整显示，停止回填")
                return "fatal"

            # 预计算每行顶线 y_top
            row_y_tops = {}
            for row in range(first_row, old_boundary):
                line_idx = row - first_row
                if line_idx < 0 or line_idx + 1 >= len(hy):
                    _print(f"[错误] 行 {row + 1} 不是完整可见行，停止回填")
                    return "fatal"
                row_y_tops[row] = float(hy[line_idx])

            # 先整体校验, 全部通过才点击(避免填一半再遇到错位)
            if verify is not None:
                for row in range(old_boundary - 1, first_row - 1, -1):
                    status, matched, conflicts, flags = verify(
                        row, row_y_tops[row])
                    if status == "fatal":
                        _print(f"[错误] 行 {row + 1} 内容与题目不一致"
                               f"(匹配 {matched} 冲突 {conflicts} 旗 {flags}), "
                               "疑似定位错位, 停止回填")
                        return "fatal"
                    if status == "continue":
                        return "continue"

            for row in range(old_boundary - 1, first_row - 1, -1):
                y_top = row_y_tops[row]
                y = y_top + pitch / 2.0
                for col in mines_by_row.get(row, []):
                    if stop.stopped:
                        _print("[中断] 回填阶段被用户中断")
                        return "fatal"
                    if col < 0 or col + 1 >= len(vx):
                        _print(f"[错误] 行 {row + 1} 列 {col + 1} 坐标越界")
                        return "fatal"
                    x = (float(vx[col]) + float(vx[col + 1])) / 2.0
                    sx, sy = left + x, top + y
                    if debug_cb:
                        debug_cb(row, col, sx, sy)
                    right_click(sx, sy, jitter=config.get("jitter", 1.5))
                    filled += 1
                    time.sleep(config.get("click_interval", 0.05))
            return "ok"

        def verify_row(row, y_top):
            """点击前校验: 未作答行数字应与原题一致; 错位时停止, 防止错旗.

            img 取自外层闭包, 在调用 click_new_rows 前已被更新为当前屏。
            返回 (status, matched, conflicts, flags)。
            """
            expected = st.global_rows.get(row)
            if expected is None:
                return "ok", 0, 0, 0
            ref_vx = st.reference_vx or gcur.get("vx") or []
            if len(ref_vx) < 2:
                return "ok", 0, 0, 0
            return _verify_row_against_board(
                img, y_top, ref_vx, pitch, expected)

        # 第一屏仍只定位一次。识别结束时停在棋盘底部，所以这屏必须连续
        # 覆盖到第 n 行；随后把可见区作为第一个整行批次处理。
        initial_visible = max(0, len(gcur.get("hy", [])) - 1)
        if g0_cur > 0 and g0_cur + initial_visible < n:
            _print("[错误] 初始底屏没有完整覆盖棋盘底端，停止回填")
            return 0
        if click_new_rows(g0_cur, completed_from, gcur, verify=verify_row) != "ok":
            return filled
        completed_from = g0_cur

        while completed_from > 0 and not stop.stopped:
            if scroll_count >= max_scrolls:
                _print(f"[错误] 向上续行超过 {max_scrolls} 次滚动，停止回填")
                return filled

            # 点击后的新截图作为相邻帧锚点。它和滚动后的画面拥有完全相同
            # 的旗子状态，图像对齐不会再受“原题/答案状态不同”影响。
            anchor_img, anchor_detected = _grab_analyze(bbox)
            if anchor_img is None or anchor_detected is None:
                _print("[错误] 无法读取已完成行画面，停止回填")
                return filled
            anchor_grid = _use_reference_columns(
                anchor_detected, st.reference_vx or anchor_detected["vx"],
                n, pitch)
            next_g0 = None
            next_img = None
            next_grid = None
            scroll_grid = anchor_grid
            seek_start_scroll = scroll_count
            no_move_streak = 0

            # 每次滚动后都从当前截图重新检测精确横线，并用“未作答数字位置 +
            # 已完成区旗帜位置”建立绝对行号。
            while next_g0 is None and scroll_count < max_scrolls:
                focus_current_board(scroll_grid)
                scroll_wheel(config.get("scroll_up_amount", 30))
                scroll_count += 1
                time.sleep(config.get("scroll_settle", 0.2))
                cur_img, cur_detected = _grab_analyze(bbox)
                if cur_img is None or cur_detected is None:
                    continue
                cur_grid = _use_reference_columns(
                    cur_detected, st.reference_vx or cur_detected["vx"],
                    n, pitch)
                if not anchor_grid.get("hy") or not cur_grid.get("hy"):
                    continue

                # 明确拒绝“滚轮没有作用却凭内容猜出新行”的情况。下一轮会基于
                # 新检测到的棋盘中心再次聚焦后重试。
                if _screens_identical(anchor_img, cur_img, cur_grid):
                    no_move_streak += 1
                    scroll_grid = cur_grid
                    if no_move_streak >= 3:
                        break
                    continue
                no_move_streak = 0

                try:
                    cur_values, _, _ = classify_screen(
                        cur_img, cur_grid, templates=config.get("templates"))
                except Exception:
                    cur_values = None

                state_g0, _ = _choose_bottomup_fill_window(
                    cur_values, st.global_rows, n, completed_from, mines_by_row)
                if state_g0 is not None:
                    next_g0 = state_g0
                    next_img, next_grid = cur_img, cur_grid
                    break
                scroll_grid = cur_grid

                # 若多次重新聚焦后仍没有绝对内容证据，停止而不让待处理边界滚出
                # 当前屏。这里按尝试次数而非滚轮像素/固定行数判断。
                seek_limit = max(4, min(10, max(1, len(cur_grid["hy"]) - 1) // 3))
                if scroll_count - seek_start_scroll >= seek_limit:
                    break

            if next_g0 is None or next_grid is None:
                _print("[错误] 无法找到新的未完成整行，停止回填")
                return filled
            if next_g0 < 0 or next_g0 >= completed_from:
                _print("[错误] 新行边界不连续，停止回填")
                return filled
            img = next_img   # 校验/点击基于滚动后的当前屏
            fill_status = click_new_rows(
                next_g0, completed_from, next_grid, verify=verify_row)
            if fill_status == "fatal":
                return filled
            if fill_status == "continue":
                # 新行尚未完整露出(顶边被视口裁掉), 本轮一个都不点; 清空定位,
                # 回到内层滚动循环继续向上, 直到该行完整进入视口。
                next_g0 = None
                next_img = None
                next_grid = None
                continue

            completed_from = next_g0
            g0_cur = next_g0
            gcur = next_grid
            img = next_img

        if stop.stopped:
            _print("[中断] 回填阶段被用户中断")
        return filled

    # 按行号从底到顶处理(识别后画面在底端)
    for r_target in target_rows:
        if stop.stopped:
            _print("[中断] 回填阶段被用户中断")
            break

        # 滚动到目标行: 若目标行不在当前屏, 根据方向滚动
        for step in range(config.get("max_fill_scroll_steps", 50)):
            if stop.stopped:
                break
            if r_target >= g0_cur and r_target < g0_cur + vals.shape[0]:
                break  # 目标行在当前屏
            # 决定滚动方向
            if r_target > g0_cur + vals.shape[0] - 1:
                amount = config.get("scroll_amount", -60)   # 向下滚
            else:
                amount = config.get("scroll_up_amount", 30)  # 向上滚
            scroll_wheel(amount)
            time.sleep(config.get("scroll_settle", 0.2))
            img, gcur = _grab_analyze(bbox)
            g0_cur, vals, gcur, img = locate_current(img, gcur)
            if g0_cur is None:
                _print(f"[错误] 滚动后无法可靠定位棋盘, 在行 {r_target + 1} 前停止")
                return filled

        if g0_cur is None or vals is None:
            _print(f"[错误] 行 {r_target + 1} 无法定位, 停止回填")
            return filled
        line_idx = r_target - g0_cur
        if line_idx < 0 or line_idx >= vals.shape[0]:
            _print(f"[错误] 行 {r_target + 1} 在滚动上限内仍未进入画面, 停止回填")
            return filled

        # 标记该行所有雷
        for c in mines_by_row[r_target]:
            if stop.stopped:
                break
            y = gcur["hy"][line_idx] + pitch / 2.0
            x = gcur["vx"][c] + pitch / 2.0
            sx = left + x
            sy = top + y
            if debug_cb:
                debug_cb(r_target, c, sx, sy)
            right_click(sx, sy, jitter=config.get("jitter", 1.5))
            filled += 1
            time.sleep(config.get("click_interval", 0.05))

    return filled


def _grab_analyze(bbox):
    img = grab_screen(bbox)
    if img is None:
        return None, None
    grid = analyze_screen(img)
    return img, grid


# ----------------------------------------------------------------------
# 框选模式回填: 框内直接插旗(全程不滚动)
# ----------------------------------------------------------------------

def fill_visible_board(st, bbox, stop, config, mine_cells, debug_cb=None):
    """框选模式回填: 全部格可见, 直接右键插旗, 全程不滚动.

    mine_cells 为求解得到的全部雷(含识别时已插旗的格)。已插旗格不重复
    点击(右键会取消旗子), 但计入完成数。每行点击前重新截屏校验该行:
    数字与原题一致、且不出现"不该有旗处的旗", 否则停止回填(不提交)。
    返回完成数(识别时已插旗 + 本次点击)。
    """
    left, top, right, bottom = bbox
    n = st.n
    pitch = st.pitch
    ref_vx = st.reference_vx or []
    if len(ref_vx) < n + 1:
        _print("[错误] 回填列线缺失, 停止回填")
        return 0

    mines_by_row = {}
    for r, c in mine_cells:
        mines_by_row.setdefault(r, []).append(c)

    already = [(r, c) for (r, c) in mine_cells
               if st.board is not None and st.board[r, c] == FLAG]
    if already:
        _print(f"[回填] 识别时已插旗 {len(already)} 格, 不重复点击")

    def verify_row(img, row, hy):
        """点击前校验: 未作答格数字与原题一致; 不允许"不该有旗处有旗"."""
        expected = st.global_rows.get(row)
        if expected is None:
            return True, 0, 0
        y_top = float(hy[row])
        cur = _row_values_at(img, y_top, ref_vx, pitch)
        if cur is None:
            return False, 0, 0
        row_mine_cols = [c for (r, c) in mine_cells if r == row]
        ev = _fill_row_evidence(cur, expected, row_mine_cols, completed=True)
        if ev is None:
            return False, 0, 0
        matched = int(ev["exact"] + ev["gray_matches"])
        if (ev["conflicts"] > max(2, matched // 4)
                or ev["flag_conflicts"] > 0):
            return False, matched, int(ev["conflicts"])
        return True, matched, int(ev["conflicts"])

    # 行位置以回填前的当前截图为准(页面静止, 与识别阶段同一画面)
    img0, grid0 = _grab_analyze(bbox)
    if img0 is None or grid0 is None:
        _print("[错误] 回填初始截屏失败")
        return 0
    grid0 = _use_reference_columns(grid0, st.reference_vx or grid0["vx"],
                                   n, pitch)
    hy = grid0.get("hy") or []
    if len(hy) - 1 < n:
        _print(f"[错误] 回填截屏仅见 {max(0, len(hy) - 1)}/{n} 行, 停止回填")
        return 0

    filled = len(already)
    clicked = 0
    for row in sorted(mines_by_row.keys()):
        if stop.stopped:
            _print("[中断] 回填阶段被用户中断")
            return filled
        # 每行点击前重新截屏校验(无滚动; 防页面渲染意外导致错位)
        img_r, _ = _grab_analyze(bbox)
        if img_r is None:
            _print(f"[错误] 行 {row + 1} 回填前截屏失败, 停止回填")
            return filled
        ok, matched, conflicts = verify_row(img_r, row, hy)
        if not ok:
            _print(f"[错误] 行 {row + 1} 点击前校验未通过"
                   f"(匹配 {matched} 冲突 {conflicts}), 疑似画面变动, 停止回填")
            return filled
        y = float(hy[row]) + pitch / 2.0
        for col in mines_by_row[row]:
            if st.board is not None and st.board[row, col] == FLAG:
                continue                      # 已插旗, 不重复点击
            if col < 0 or col + 1 >= len(ref_vx):
                _print(f"[错误] 行 {row + 1} 列 {col + 1} 坐标越界")
                return filled
            x = (float(ref_vx[col]) + float(ref_vx[col + 1])) / 2.0
            sx, sy = left + x, top + y
            if debug_cb:
                debug_cb(row, col, sx, sy)
            right_click(sx, sy, jitter=config.get("jitter", 1.5))
            filled += 1
            clicked += 1
            time.sleep(config.get("click_interval", 0.05))

    _print(f"[回填] 完成: 本次点击 {clicked} 格, 含识别时已插旗共 {filled} 格, "
           "全程未滚动")
    return filled
