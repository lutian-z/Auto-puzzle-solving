# -*- coding: utf-8 -*-
"""automation.py — 帐篷项目屏幕交互与键鼠作答.

职责: DPI 感知 / 全屏覆盖层框选 / 屏幕截图 / 锚点校准 / 瞬移点击作答 /
回车提交 / ESC 急停. 依赖 tkinter、mss、pyautogui、keyboard(仅实机流程
导入, 离线自测不需要).

可移植性: 本模块不含任何硬编码路径; 全部行为参数来自 config(cfg dict).
键鼠安全约定:
    - 点击一律使用 pyautogui.click(x, y) 坐标瞬移(不播放移动动画),
      不调用任何平滑移动接口;
    - 作答过程中随时可按 ESC 中断(StopFlag 热键), 中断后不再点击;
    - pyautogui.FAILSAFE 保持默认开启: 鼠标被手动推到屏幕左上角会
      触发保护性中止, 作为最后一道保险.
"""
import random
import sys
import threading
import time

IS_WINDOWS = sys.platform.startswith("win")


class InputSimulationError(Exception):
    """模拟键鼠失败(坐标越界/截图失败等)."""


class StopRequested(Exception):
    """用户按 ESC 请求中断作答."""


class StopFlag:
    """ESC 急停旗标(沿用数独/数墙项目的热键中断模式)."""

    def __init__(self):
        self._stop = threading.Event()
        self._hooked = False

    def start(self, hotkey="esc"):
        try:
            import keyboard
            keyboard.add_hotkey(hotkey, self._stop.set)
            self._hooked = True
            return True
        except Exception:
            # 无 keyboard 库或无权限时不阻塞主流程, 退化为主循环里
            # 检查不到热键, 仅能靠 Ctrl+C 中断.
            return False

    @property
    def stopped(self):
        return self._stop.is_set()

    def cleanup(self):
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass


def set_dpi_aware():
    """Windows 高 DPI 屏: 让进程以物理像素工作, 截图与点击坐标一致."""
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _physical_scaling():
    """tkinter 逻辑像素 → 物理像素 的换算比例(截图用物理像素)."""
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
    """半透明全屏覆盖层, 用户从左上拖到右下框选帐篷题目.

    返回物理像素坐标 ((x0,y0), (x1,y1)); 框选可稍大于题目(含四周数字),
    识别阶段自会定位网格并裁剪. ESC 取消返回 None.
    """
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

    state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "rect": None,
             "done": False, "cancel": False}

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
            w = abs(event.x_root - state["x0"])
            h = abs(event.y_root - state["y0"])
            canvas.itemconfig(state["size_text"],
                              text=f"框选尺寸: {w} x {h}  (框住棋盘和四周数字即可, 可略大)")
            canvas.coords(state["size_text"], 10, 60)

    def on_release(event):
        state["done"] = True
        state["x1"], state["y1"] = event.x_root, event.y_root

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: (state.__setitem__("cancel", True),
                                     state.__setitem__("done", True)))
    root.bind("<Return>", lambda e: state.__setitem__("done", True))

    canvas.create_text(
        10, 10, anchor="nw", fill="white",
        font=("Microsoft YaHei", 16) if IS_WINDOWS else ("DejaVu Sans", 16),
        text="拖动鼠标框选整个帐篷棋盘(含上方与左侧数字)\n松开完成, ESC 取消")
    state["size_text"] = canvas.create_text(
        10, 60, anchor="nw", fill="yellow",
        font=("Microsoft YaHei", 13) if IS_WINDOWS else ("DejaVu Sans", 13),
        text="框选尺寸: -")

    while not state["done"]:
        root.update()
        time.sleep(0.02)
    root.destroy()

    if state["cancel"]:
        return None
    x0, y0 = min(state["x0"], state["x1"]), min(state["y0"], state["y1"])
    x1, y1 = max(state["x0"], state["x1"]), max(state["y0"], state["y1"])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        print(f"[DPI] 逻辑像素换算比例 x={sx:.2f} y={sy:.2f}")
    return (int(round(x0 * sx)), int(round(y0 * sy))), \
        (int(round(x1 * sx)), int(round(y1 * sy)))


def grab_screen(bbox):
    """截取屏幕区域 bbox((x0,y0),(x1,y1), 物理像素), 返回 BGR ndarray."""
    try:
        import mss
        import numpy as np
        import cv2
        left, top = bbox[0]
        right, bottom = bbox[1]
        if right - left < 10 or bottom - top < 10:
            raise InputSimulationError(f"截图区域过小: {bbox}")
        monitor = {"left": left, "top": top,
                   "width": right - left, "height": bottom - top}
        with mss.MSS() as sct:
            shot = sct.grab(monitor)
        img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
            shot.height, shot.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except InputSimulationError:
        raise
    except Exception as e:
        raise InputSimulationError(f"截图失败: {e}")


def calibrate_anchor(bbox, puzzle, cfg, log):
    """作答前锚点校准: 重新截图定位棋盘外接框, 检测窗口是否移动.

    bbox: 原框选区域; puzzle.board_origin 为棋盘外接框左上角在原截图
    坐标系中的位置. 校准截图向四周扩大 calib_margin px(窗口移动后棋盘
    仍能被定位). 识别经过缩放/转正时 board_origin 与重定位结果不可比,
    跳过校准并告警. 定位失败保留原值.
    """
    if abs(getattr(puzzle, "scale", 1.0) - 1.0) > 1e-6 or \
            abs(getattr(puzzle, "rot_ang", 0.0)) > 1e-6:
        log.warning("[校准] 识别经过缩放/转正, 锚点校准不可比, 跳过")
        return bbox[0]
    import recognizer
    margin = int(cfg["calib_margin"])
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    try:
        import mss
        with mss.MSS() as sct:
            mon = sct.monitors[0]
        X0 = max(int(mon["left"]), x0 - margin)
        Y0 = max(int(mon["top"]), y0 - margin)
        X1 = min(int(mon["left"]) + int(mon["width"]), x1 + margin)
        Y1 = min(int(mon["top"]) + int(mon["height"]), y1 + margin)
    except Exception:
        X0, Y0, X1, Y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    try:
        img = grab_screen(((X0, Y0), (X1, Y1)))
        nb = recognizer.find_board_bbox(img, cfg)
    except Exception as e:
        log.warning("[校准] 锚点校准失败(%s), 沿用原坐标", e)
        return bbox[0]
    return calibrate_shift(nb, (X0, Y0), puzzle, bbox[0], log)


def calibrate_shift(nb, expanded_origin, puzzle, fallback, log):
    """锚点校准的纯几何部分(可离线测试).

    nb: 扩大截图中棋盘外接框; expanded_origin: 扩大截图的屏幕原点;
    puzzle.board_origin: 棋盘外接框在"原 bbox 截图"中的位置——比较前
    必须加上 bbox 原点换算成屏幕绝对坐标(此前漏加, 会把框选原点整个
    当成位移, 框选不在屏幕原点时校准必然误报移动并打飞点击基准).
    偏移 >3px 返回平移后的原点, 否则返回 fallback.
    """
    ox0 = fallback[0] + float(puzzle.board_origin[0])
    oy0 = fallback[1] + float(puzzle.board_origin[1])
    dx = float(nb[0] + expanded_origin[0]) - ox0
    dy = float(nb[1] + expanded_origin[1]) - oy0
    if abs(dx) <= 3 and abs(dy) <= 3:
        return fallback
    log.info("[校准] 检测到棋盘移动 (%+.0f,%+.0f)px, 已校正点击基准", dx, dy)
    return (int(round(fallback[0] + dx)), int(round(fallback[1] + dy)))


def plan_clicks(puzzle, cells, bbox_origin):
    """把一组格子转换为屏幕像素坐标列表(行优先). cells: 可迭代的 (r,c).

    puzzle.centers 为识别输入图坐标, 加上框选原点即屏幕物理像素.
    """
    ox, oy = bbox_origin
    plan = []
    for (r, c) in sorted(cells):
        cx, cy = puzzle.centers[(r, c)]
        plan.append((int(round(ox + cx)), int(round(oy + cy))))
    return plan


def fill_answer(plan, cfg, stop=None, log=None):
    """按 plan 瞬移点击. 返回点击格数; stop 置位时抛 StopRequested."""
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.MINIMUM_DURATION = 0.0
    pyautogui.MINIMUM_SLEEP = 0.0

    screen_w, screen_h = pyautogui.size()
    jitter = cfg["jitter"]
    click_interval = cfg["click_interval"]
    cell_delay = cfg["cell_delay"]

    clicked = 0
    for (tx, ty) in plan:
        if stop is not None and stop.stopped:
            raise StopRequested()
        if not (0 <= tx <= screen_w and 0 <= ty <= screen_h):
            raise InputSimulationError(
                f"点击坐标越界: ({tx},{ty}), 请确认题目完整显示在屏幕上")
        jx = random.uniform(-jitter, jitter)
        jy = random.uniform(-jitter, jitter)
        # 坐标瞬移点击(无移动动画)
        pyautogui.click(int(tx + jx), int(ty + jy))
        clicked += 1
        time.sleep(click_interval)
        if clicked % 20 == 0:
            time.sleep(cell_delay)
    time.sleep(cell_delay)
    return clicked


def submit_answer(cfg, log):
    """按提交键(默认回车)."""
    import pyautogui
    pyautogui.PAUSE = 0.0
    key = cfg["submit_hotkey"]
    pyautogui.press(key)
    log.info("[作答] 已按 %s 提交", key)
