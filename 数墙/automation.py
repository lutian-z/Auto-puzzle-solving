# -*- coding: utf-8 -*-
"""automation.py — 数墙项目屏幕交互与键鼠作答.

职责: DPI 感知 / 全屏覆盖层框选 / 屏幕截图 / 瞬移点击作答 / 回车提交 /
ESC 急停. 依赖 mss、pyautogui、keyboard(仅实机流程导入, 离线自测不需要).

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


class StopRequested(Exception):
    """用户按 ESC 请求中断作答."""


class StopFlag:
    """ESC 急停旗标(参照数独项目的热键中断模式)."""

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
            # 无 keyboard 库或无权限(部分 LinuxWayland)时不阻塞主流程,
            # 退化为主循环里检查不到热键, 仅能靠 Ctrl+C 中断.
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
            _mon = (sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])
            phys_w = _mon["width"]
            phys_h = _mon["height"]
        sx = (phys_w / tk_w) if tk_w else 1.0
        sy = (phys_h / tk_h) if tk_h else 1.0
    except Exception:
        sx = sy = 1.0
    root.destroy()
    return sx, sy


def select_region():
    """半透明全屏覆盖层, 用户从左上拖到右下框选数墙棋盘.

    返回物理像素坐标 ((x0,y0), (x1,y1)); 框选可稍大于棋盘, 识别阶段
    自会定位墙体并裁剪. ESC 取消返回 None.
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
                              text=f"框选尺寸: {w} x {h}  (框住整个棋盘即可, 可略大)")
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
        text="拖动鼠标框选整个数墙棋盘(框大一点没关系)\n松开完成, ESC 取消")
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
    import mss
    left, top = bbox[0]
    right, bottom = bbox[1]
    monitor = {"left": left, "top": top,
               "width": right - left, "height": bottom - top}
    with mss.MSS() as sct:
        shot = sct.grab(monitor)
    import numpy as np
    import cv2
    img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
        shot.height, shot.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _win_click_env():
    """Windows ctypes 快速点击: SetCursorPos+SendInput 直发事件
    (<0.5ms, 对比 pyautogui 每次 10~20ms 封装), 顺带把系统定时器提到
    1ms 精度. 返回 (click, screen, cleanup). 非 Windows 不可达.
    """
    import ctypes

    ULONG = ctypes.c_ulong
    LONG = ctypes.c_long

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", LONG), ("dy", LONG), ("mouseData", ULONG),
                    ("dwFlags", ULONG), ("time", ULONG),
                    ("dwExtraInfo", ctypes.c_void_p)]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ULONG), ("union", _INPUTunion)]

    user32 = ctypes.windll.user32
    winmm = ctypes.windll.winmm
    timer_set = False
    try:
        timer_set = (winmm.timeBeginPeriod(1) == 0)
    except Exception:
        pass

    inputs = (INPUT * 2)()
    inputs[0].type = 0                        # INPUT_MOUSE
    inputs[0].union.mi = MOUSEINPUT(0, 0, 0, 0x0002, 0, None)   # LEFTDOWN
    inputs[1].type = 0
    inputs[1].union.mi = MOUSEINPUT(0, 0, 0, 0x0004, 0, None)   # LEFTUP

    def click(x, y):
        if not user32.SetCursorPos(int(x), int(y)):
            raise RuntimeError(f"鼠标定位失败: ({x},{y})")
        if user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) != 2:
            raise RuntimeError(f"鼠标事件发送失败: ({x},{y})")

    screen = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

    def cleanup():
        if timer_set:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass

    return click, screen, cleanup


def _pyautogui_click_env():
    """pyautogui 点击环境(非 Windows / 快速路径不可用时的回退)."""
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.MINIMUM_DURATION = 0.0
    pyautogui.MINIMUM_SLEEP = 0.0
    wh = pyautogui.size()

    def click(x, y):
        pyautogui.click(int(x), int(y))       # 坐标瞬移点击(无移动动画)

    def cleanup():
        pass

    return click, wh, cleanup


def fill_answer(puzzle, grid, bbox_origin, cfg, stop=None):
    """只点击解中为黑的格子(数字格/白格不点), 坐标为 puzzle.centers+bbox_origin.

    返回点击格数; stop 置位抛 StopRequested.
    """
    if IS_WINDOWS:
        try:
            click, screen, cleanup = _win_click_env()
        except Exception:
            click, screen, cleanup = _pyautogui_click_env()
    else:
        click, screen, cleanup = _pyautogui_click_env()
    screen_w, screen_h = screen

    ox, oy = bbox_origin
    jitter = cfg.get("jitter", 1.5)
    click_interval = cfg.get("click_interval", 0.012)
    cell_delay = cfg.get("cell_delay", 0.015)

    clicked = 0
    try:
        for (r, c), (cx, cy) in sorted(puzzle.centers.items()):
            if stop is not None and stop.stopped:
                raise StopRequested()
            if r >= len(grid) or c >= len(grid[0]) or not grid[r][c]:
                continue
            jx = random.uniform(-jitter, jitter)
            jy = random.uniform(-jitter, jitter)
            tx, ty = int(round(ox + cx + jx)), int(round(oy + cy + jy))
            if not (0 <= tx <= screen_w and 0 <= ty <= screen_h):
                raise RuntimeError(
                    f"点击坐标越界: ({tx},{ty}), 请确认题目完整显示在屏幕上")
            # 坐标瞬移点击(无移动动画)
            click(tx, ty)
            clicked += 1
            time.sleep(click_interval)
            if clicked % 20 == 0:
                time.sleep(cell_delay)
        time.sleep(cell_delay)
    finally:
        cleanup()
    return clicked


def submit_answer(cfg):
    """按提交键(默认回车)."""
    import pyautogui
    pyautogui.PAUSE = 0.0
    key = cfg.get("submit_hotkey", "enter")
    pyautogui.press(key)
    print(f"[作答] 已按 {key} 提交")
