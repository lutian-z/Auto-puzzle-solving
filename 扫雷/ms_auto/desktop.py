# -*- coding: utf-8 -*-
"""ms_auto.desktop — 桌面自动化层: 中断热键 / 选题 / 截屏 / 鼠标键盘
"""
import sys
import time
import threading

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


# 区域选择(拖动框选 / 点击两个点, 按鼠标手势自动识别)

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
            _mon = (sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])
            phys_w = _mon["width"]
            phys_h = _mon["height"]
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


# 截屏

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


# 鼠标操作

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


_WIN_RIGHT = None      # 缓存: 可调用=快速路径可用, False=回退 pyautogui


def _win_right_clicker():
    """Windows ctypes 右键快速路径(借鉴马赛克项目实测结论).

    pyautogui 单次 click 含坐标规整/FAILSAFE/平台封装约 10~20ms, 几百次
    插旗即 10 秒级; SetCursorPos+SendInput 直发事件 <0.5ms, 瞬移语义相同。
    同时 winmm.timeBeginPeriod(1) 把系统定时器精度从默认 15.6ms 提到 1ms,
    否则 time.sleep(0.006) 实际睡到 ~15.6ms, 间隔再小也白搭。
    """
    global _WIN_RIGHT
    if _WIN_RIGHT is None:
        if not IS_WINDOWS:
            _WIN_RIGHT = False
        else:
            try:
                import ctypes

                class MOUSEINPUT(ctypes.Structure):
                    _fields_ = [("dx", ctypes.c_long),
                                ("dy", ctypes.c_long),
                                ("mouseData", ctypes.c_ulong),
                                ("dwFlags", ctypes.c_ulong),
                                ("time", ctypes.c_ulong),
                                ("dwExtraInfo", ctypes.c_void_p)]

                class _U(ctypes.Union):
                    _fields_ = [("mi", MOUSEINPUT)]

                class INPUT(ctypes.Structure):
                    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]

                user32 = ctypes.windll.user32
                try:
                    ctypes.windll.winmm.timeBeginPeriod(1)
                except Exception:
                    pass
                inputs = (INPUT * 2)()
                inputs[0].type = 0
                inputs[0].u.mi = MOUSEINPUT(0, 0, 0, 0x0008, 0, None)  # RBUTTONDOWN
                inputs[1].type = 0
                inputs[1].u.mi = MOUSEINPUT(0, 0, 0, 0x0010, 0, None)  # RBUTTONUP

                def click_right(x, y):
                    if not user32.SetCursorPos(int(x), int(y)):
                        raise RuntimeError("鼠标定位失败: (%d,%d)" % (x, y))
                    if user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) != 2:
                        raise RuntimeError("右键事件发送失败: (%d,%d)" % (x, y))

                _WIN_RIGHT = click_right
            except Exception:
                _WIN_RIGHT = False
    return _WIN_RIGHT


def right_click(x, y, jitter=1.5):
    """在 (x,y) 附近右键点击(标记雷), 带随机抖动(避免点到网格线上)."""
    import random
    jx = random.uniform(-jitter, jitter)
    jy = random.uniform(-jitter, jitter)
    tx, ty = int(round(x + jx)), int(round(y + jy))
    fast = _win_right_clicker()
    if fast is not False:
        try:
            fast(tx, ty)
            return
        except Exception:
            pass                      # 快速路径异常 -> 回退 pyautogui 重试
    _pg().click(x=tx, y=ty, button="right")


def press_enter():
    _pg().press("enter")




def _print(msg):
    print(msg, flush=True)

