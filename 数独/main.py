import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict

import cv2
import keyboard
import math
import mss
import numpy as np
import pyautogui
from PIL import Image, ImageDraw, ImageFont

IS_WINDOWS = sys.platform.startswith("win")

CONFIG = {
    "stop_hotkey": "esc",
    "typewrite_interval": 0.02,   # 已改 press 单键直发, 此项保留兼容
    "click_interval": 0.015,
    "cell_delay": 0.02,
    "jitter": 1.5,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_SIZE = 28

MIN_SIZE = 4
MAX_SIZE = 16
SYMBOLS = "123456789ABCDEFG"

# 28x28 符号模板(内嵌), 数字 1-9 + 字母 A-G
EMBEDDED_TEMPLATE_B64 = """HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWftzRcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqh++glAAAAAAAAAAAAAAAAAAAAAAAAAAAABiCO6/3oJQAAAAAAAAAAAAAAAAAAAAAAAAAAdLfX9///6CUAAAAAAAAAAAAAAAAAAAAAAAAAAMn//////+glAAAAAAAAAAAAAAAAAAAAAAAAAAA6bXil8f/oJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGeD/6CUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABHf/+glAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR3//oJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEd//6CUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABHf/+glAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR3//oJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEd//6CUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABHf/+glAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR3//oJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEd//6CUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABHf/+glAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR3//oJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFN//6SgAAAAAAAAAAAAAAAAAAAAAAAAAAD5wdZ7x//mmdXJRAAAAAAAAAAAAAAAAAAAAAADG9vf7/v///ff24wAAAAAAAAAAAAAAAAAAAAAAyev8/////////NMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARdSuev7//euNgkAAAAAAAAAAAAAAAAAAAAAADHY/P///////fKSFAEAAAAAAAAAAAAAAAAAAABy///rupWYyv//9owTAAAAAAAAAAAAAAAAAAAAWOKrSRACAiWi+//lRgIAAAAAAAAAAAAAAAAAABM/CAAAAAAAMb7/9o0GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+C//agCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMe//2qgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGpP/9ZEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAADzD/+tLAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAF+9//BKAEAAAAAAAAAAAAAAAAAAAAAAAAAAAI+0v/ydgsAAAAAAAAAAAAAAAAAAAAAAAAAAAATsf/+sx8BAAAAAAAAAAAAAAAAAAAAAAAAAAAPgfr/zTICAAAAAAAAAAAAAAAAAAAAAAAAAAAJdOv/1kUDAAAAAAAAAAAAAAAAAAAAAAAAAAAJbev+8VYCAAAAAAAAAAAAAAAAAAAAAAAAAAAFbur/5l4AAAAAAAAAAAAAAAAAAAAAAAAAAAAGXOb/4VgAAAAAAAAAAAAAAAAAAAAAAAAAAAADaej/3U0CAAAAAAAAAAAAAAAAAAAAAAAAAAAAVe3/408EAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOX//9yfkJCQkJCQjoI1AAAAAAAAAAAAAAAAAAD///////////////77gAAAAAAAAAAAAAAAAAAA1/z////////////84IgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQjacrx/P/83YYqDAAAAAAAAAAAAAAAAAAAAAA/7vz////////92m0PAAAAAAAAAAAAAAAAAAAAbf7537ial6rv///STQQAAAAAAAAAAAAAAAAAAEq/kjYIAAAOZMH/+ncgAAAAAAAAAAAAAAAAAAAJGgAAAAAAAA107v+3IgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU+n/0CIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFjy/rIcAAAAAAAAAAAAAAAAAAAAAAAAAAAAACu3//NUAAAAAAAAAAAAAAAAAAAAAAAAAAwiLVC9/Ot0CwAAAAAAAAAAAAAAAAAAAAAAAABRtNr3/+NmCQAAAAAAAAAAAAAAAAAAAAAAAAAAiP3////bgCoCAAAAAAAAAAAAAAAAAAAAAAAAAEONmrHi9+6aMwIAAAAAAAAAAAAAAAAAAAAAAAAIFhkcVLP474UgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZYz//VPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOK3/+E4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAADOm//tOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABEvv/yTgAAAAAAAAAAAAAAAAAARAgAAAAAAAANbe7/1zgAAAAAAAAAAAAAAAAAANqvaxwAAAAYa8//+4IXAAAAAAAAAAAAAAAAAAD///bLoI6QsPL//680AAAAAAAAAAAAAAAAAAAA2vr/////////+7E9AAAAAAAAAAAAAAAAAAAAABU7ifL////7w1ohAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEXno/ocPAAAAAAAAAAAAAAAAAAAAAAAAAAAABnL0//+mHgAAAAAAAAAAAAAAAAAAAAAAAAAABWTl////ph4AAAAAAAAAAAAAAAAAAAAAAAAABlXd/////6YeAAAAAAAAAAAAAAAAAAAAAAAAAEHM/NKj7v+mHgAAAAAAAAAAAAAAAAAAAAAAADy9/OBQPuT/ph4AAAAAAAAAAAAAAAAAAAAAAC+/+uRUBT3k/6YeAAAAAAAAAAAAAAAAAAAAABqr/OBWBQA95P+mHgAAAAAAAAAAAAAAAAAAABqo/OJoCAAAPeT/ph4AAAAAAAAAAAAAAAAAABaO+PR4CwAAAD3k/6YeAAAAAAAAAAAAAAAAAAB05vaRFQAAAABA5f+mHgAAAAAAAAAAAAAAAAAAyv/dh1dPT09RmfP/zoFKDAAAAAAAAAAAAAAAANr/+/Pz8/Pz9Pr///v181EAAAAAAAAAAAAAAADV8/v7+/v7+/v7/v//+/ZTAAAAAAAAAAAAAAAALkJHR0dHR0dKh/b/yGI5FwAAAAAAAAAAAAAAAAAAAAAAAAAAAD3k/6YeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA95P+mHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPeT/ph4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD3k/6YeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA95P+mHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPeT/ph4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACrE8YsYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGfP/////////////2xUAAAAAAAAAAAAAAAAAABn4///9+/v7+/v7++gRAAAAAAAAAAAAAAAAAAAZ+P/vl39/f39/f39fAAAAAAAAAAAAAAAAAAAAGfj/wxEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABn4/70OAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZ+P+9DgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGfj/vQ4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAABn4/8YvGxsQBgAAAAAAAAAAAAAAAAAAAAAAAAAZ+P/00MPDvJNGDwAAAAAAAAAAAAAAAAAAAAAAGfj////////64KUeAAAAAAAAAAAAAAAAAAAAABLQ0banp63g/P/wsi4BAAAAAAAAAAAAAAAAAAAIS0MpJSUlNon5/vh1AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAIbgvn/zRUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT7r/fgVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAt0Pr/FQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMNn8+xUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFT3/+UVAAAAAAAAAAAAAAAAAAAxCQAAAAAAABWd+f+UCQAAAAAAAAAAAAAAAAAA1LVgHwAAAByI8f3pNwEAAAAAAAAAAAAAAAAAAP/+89KfkJTG9f3/hQ4AAAAAAAAAAAAAAAAAAADc/f////7+///yjBAAAAAAAAAAAAAAAAAAAAAAE0yi8P///++YNw8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUPZro////sRoAAAAAAAAAAAAAAAAAAAAAAAoxq/L8/////+wzAAAAAAAAAAAAAAAAAAAAAAdKyP//7qyUlaifHgAAAAAAAAAAAAAAAAAAAAAotf77rFsDAAALDwEAAAAAAAAAAAAAAAAAAAAWdvn7sB0AAAAAAAAAAAAAAAAAAAAAAAAAAAAALd3/vDQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC2Lw8I0JAAAAAAAAAAAAAAAAAAAAAAAAAAAAABOr/99tCiQ4OzsqDAEAAAAAAAAAAAAAAAAAAAAaz//jl6zq8vPz55UnAwAAAAAAAAAAAAAAAAAAI+H///338/Ly9f/8wTgCAAAAAAAAAAAAAAAAADX9//zlsHpvdp3j+f/KIwAAAAAAAAAAAAAAAABD///YbRMAAAALZN7//W0AAAAAAAAAAAAAAAAAQ///tRUAAAAAAAeK+P/DAAAAAAAAAAAAAAAAAEP9/7wZAAAAAAAAUuz/7gAAAAAAAAAAAAAAAAAs6P++HAAAAAAAAEnn//8AAAAAAAAAAAAAAAAAHtb/0E4AAAAAAABK6P/zAAAAAAAAAAAAAAAAABer/ut6AwAAAAAAZvv/ygAAAAAAAAAAAAAAAAALZ/H7syoBAAAAEKX//nIAAAAAAAAAAAAAAAAAAC/G//KkKQoCIIzw/98oAAAAAAAAAAAAAAAAAAANVNr/+82dlb73//JoAgAAAAAAAAAAAAAAAAAAABFk1/v//////91hCQAAAAAAAAAAAAAAAAAAAAAACCyR4/396pgpAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACfM///////////////3sgAAAAAAAAAAAAAAAAAn/f///////////////9cAAAAAAAAAAAAAAAAAGY2Tk5OTk5OTk5TD+f/RAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJZfP/pgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKrL87U8AAAAAAAAAAAAAAAAAAAAAAAAAAAAABlvq/bcXAAAAAAAAAAAAAAAAAAAAAAAAAAAAACav/e1SAQAAAAAAAAAAAAAAAAAAAAAAAAAAAARj/f6lDQAAAAAAAAAAAAAAAAAAAAAAAAAAAAEkvP/kUQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGaPz7sBMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJMr/4lcAAAAAAAAAAAAAAAAAAAAAAAAAAAAABW//+rIPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC3H/+NUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABv9/u3CQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv2P7oTwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHcfX7qwcAAAAAAAAAAAAAAAAAAAAAAAAAAAAANdT/6EkAAAAAAAAAAAAAAAAAAAAAAAAAAAAACoH6/6UHAAAAAAAAAAAAAAAAAAAAAAAAAAAAADHQ/+RMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2A+/6nEQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAn1//fTQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAIWf3/rw4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIIQ6jj9f/oqjwJAAAAAAAAAAAAAAAAAAAAAAIkl+n7/Pj4/P/wmhYAAAAAAAAAAAAAAAAAAAAQifb/5KaBg6Tk//+IBQAAAAAAAAAAAAAAAAACROb/31gKAQIKY+z/7jwAAAAAAAAAAAAAAAAABXr59poFAAAAAA2v//9lAAAAAAAAAAAAAAAAAAWG+vCPAAAAAAAGl///aQAAAAAAAAAAAAAAAAAFhfr8rhQAAAAAC57//10AAAAAAAAAAAAAAAAABGPx//iNGgIAACLT/+QrAAAAAAAAAAAAAAAAAAAQmPn/+bZZGhyB+vNlAwAAAAAAAAAAAAAAAAAAABmd+P//9sbN9O96CQAAAAAAAAAAAAAAAAAAAAAMcvb/9f3////bTAMAAAAAAAAAAAAAAAAAAAAJbuL0u3qb0vH//9E8AgAAAAAAAAAAAAAAAAADS+b/vSYAABhz3P//3jcAAAAAAAAAAAAAAAAACa755WECAAAABkLO//+nAAAAAAAAAAAAAAAAADHj/9IqAAAAAAAAX/P/5QAAAAAAAAAAAAAAAAA8/f/TJwAAAAAAADzy//8AAAAAAAAAAAAAAAAAOP7/3DUAAAAAAAA+8//3AAAAAAAAAAAAAAAAABzH/+uDBAAAAAAHb/v/wQAAAAAAAAAAAAAAAAAJe/j/2FUHAAAKUNz/9lEAAAAAAAAAAAAAAAAAASa1///ooI2KmN///5kOAAAAAAAAAAAAAAAAAAACM7Hy/////v7/7pAQAAAAAAAAAAAAAAAAAAAAAAIOTrnt/f3rojgKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACjOC1fP6wl8bAQAAAAAAAAAAAAAAAAAAAAAAGXTb+//////6wT4DAAAAAAAAAAAAAAAAAAAAEHPr//bAkKrq///ZMgAAAAAAAAAAAAAAAAAAAUPW/9x8HwgNPc///ZQGAAAAAAAAAAAAAAAAABKT+/GTCgAAAAJB4v/oRwAAAAAAAAAAAAAAAAAl3//hTQAAAAAABqP//H4AAAAAAAAAAAAAAAAAQfz/zDwAAAAAAABe7v+sAAAAAAAAAAAAAAAAAET//8o7AAAAAAAAM+j/yQAAAAAAAAAAAAAAAAA+/P/aQwAAAAAAADHo//QAAAAAAAAAAAAAAAAAGM//6noHAAAAAAI36v/6AAAAAAAAAAAAAAAAABCC/f/eaBQODhFBqfb/9QAAAAAAAAAAAAAAAAABO8L9//Gthoak4v7//+MAAAAAAAAAAAAAAAAAAAZAvvn+//////r7///AAAAAAAAAAAAAAAAAAAAABSJvwdLTz6xPje//nAAAAAAAAAAAAAAAAAAAAAAABBMjJyEEAYj3/HoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABW5//JBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJQ5f+4DQAAAAAAAAAAAAAAAAAAAAAAAAAAAAMow//vWQMAAAAAAAAAAAAAAAAAAAAJFgkGBhRKxvf3nA0AAAAAAAAAAAAAAAAAAAACQ5min5+87P/5siUDAAAAAAAAAAAAAAAAAAAABFr5/v/////tlh0DAAAAAAAAAAAAAAAAAAAAAABE3vf/+eOsPQoBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK9r/+bQUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGz+///sSAAAAAAAAAAAAAAAAAAAAAAAAAAAAAmp/////5kBAAAAAAAAAAAAAAAAAAAAAAAAAAAq6f/5///JDQAAAAAAAAAAAAAAAAAAAAAAAAAAcf//ytP/80UAAAAAAAAAAAAAAAAAAAAAAAAABLj/5E+X+/2PAAAAAAAAAAAAAAAAAAAAAAAAABrh/7gZSeH+xQsAAAAAAAAAAAAAAAAAAAAAAAOL+f6gABnC/uFRAAAAAAAAAAAAAAAAAAAAAAAQqfv0WAADn/r6hQIAAAAAAAAAAAAAAAAAAAAAKt3/3BQAAEbv/7kZAAAAAAAAAAAAAAAAAAAAAHv5/7QCAAAX0P/rTwAAAAAAAAAAAAAAAAAAAAa3//1uAAAAEIz/+oUEAAAAAAAAAAAAAAAAAAA42f/jIQAAAAVN+/+6KAAAAAAAAAAAAAAAAAABbPr/43daWVlZkv3/20wAAAAAAAAAAAAAAAAAC6L////8+/v7+/z///txCAAAAAAAAAAAAAAAAC7t///07+/v7+/x9f//yRwAAAAAAAAAAAAAAABV///JYFBQUFBQU4Dj/+07AAAAAAAAAAAAAAALpf/5ZgIAAAAAAAAUkf/wcQYAAAAAAAAAAAAAMOv/6TcAAAAAAAAAA3b+/bwZAAAAAAAAAAAAAFT//7cSAAAAAAAAAAA+0//0LgAAAAAAAAAAAACv//piAgAAAAAAAAAADp///40AAAAAAAAAAAAAyPfAKQAAAAAAAAAAAAJg1faUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACe9f/////////4wFQNAAAAAAAAAAAAAAAAAAAAsv/////////////vgBAAAAAAAAAAAAAAAAAAALL/9NCenJycncT8//RjBgAAAAAAAAAAAAAAAACy/9tuAwAAAAIpufz/whYAAAAAAAAAAAAAAAAAsv/WZwMAAAAAA1n3/+YqAQAAAAAAAAAAAAAAALL/1mcDAAAAAABH6f/uIAAAAAAAAAAAAAAAAACy/9ZnAwAAAAAAUfX/qgsAAAAAAAAAAAAAAAAAsv/WZwMAAAAABXn8/GQFAAAAAAAAAAAAAAAAALL/23YfGRkZKIvu+asfAAAAAAAAAAAAAAAAAACy//jZw7u7vNT9/8YtAAAAAAAAAAAAAAAAAAAAsv/////////////YbQwAAAAAAAAAAAAAAAAAALL/9NKqn5+gsNP9+eWZDgAAAAAAAAAAAAAAAACy/9x3ExAQEBoqi+H/8lsBAAAAAAAAAAAAAAAAsv/WZwMAAAAAAROe/f++BwAAAAAAAAAAAAAAALL/1mcDAAAAAAAAQ+j//RkAAAAAAAAAAAAAAACy/9ZnAwAAAAAAAD/o//8ZAAAAAAAAAAAAAAAAsv/WZwMAAAAAAABk8v/qGQAAAAAAAAAAAAAAALL/1mcDAAAAAAAaovn/rAsAAAAAAAAAAAAAAACy/9tuBQAAAAlEre797T0BAAAAAAAAAAAAAAAAsv/0y5KJiYmh3vz/+XoPAAAAAAAAAAAAAAAAALL///76+vr6/f/7wlwRAAAAAAAAAAAAAAAAAACe4f////////vFYBgEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBR1q0fj////5tFMeAQAAAAAAAAAAAAAAAAABF1vD7/3////////45moAAAAAAAAAAAAAAAABF2Db//ngtZSUlLfm/P//AAAAAAAAAAAAAAAADl3W/vC3SQoAAAAVRInh7AAAAAAAAAAAAAAABDyw/+avLQAAAAAAAAAASGYAAAAAAAAAAAAAABd96Pm9RAAAAAAAAAAAAAAHAAAAAAAAAAAAAAA6o/vfiBQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWcn/z1oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHf9r9GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABx9+qpEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcf/kngsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHH/5J4LAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABx9+qpEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcer2v0kAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFnJ/89iAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA6pv3fhREAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFnrq+cRJAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAQ7t//vvjkAAAAAAAAAB01jAAAAAAAAAAAAAAABFmTg/vm6VAwAAAAWUZzm6gAAAAAAAAAAAAAAAAEXaOX/+eCqf3+Gu+v8//YAAAAAAAAAAAAAAAAAARddw+/9+vr6/P//9cNkAAAAAAAAAAAAAAAAAAABBR1XvvH////oljsXAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///////////8mj8WBQAAAAAAAAAAAAAAAAAAAP////v29vb9////12YlBwAAAAAAAAAAAAAAAAD//++ugoKCosv0////yUsIAAAAAAAAAAAAAAAA///HLgAAAAAyb57j///KPQQAAAAAAAAAAAAAAP//vx4AAAAAAAAObuH//7kZAAAAAAAAAAAAAAD//78eAAAAAAAAAA5u4f//QgMAAAAAAAAAAAAA//+/HgAAAAAAAAAACpb//9YJAAAAAAAAAAAAAP//vx4AAAAAAAAAAABs5///NgAAAAAAAAAAAAD//78eAAAAAAAAAAAAB6n//0EAAAAAAAAAAAAA//+/HgAAAAAAAAAAAACl//9mAAAAAAAAAAAAAP//vx4AAAAAAAAAAAAApf//ZgAAAAAAAAAAAAD//78eAAAAAAAAAAAAAKX//2YAAAAAAAAAAAAA//+/HgAAAAAAAAAAAACl//9mAAAAAAAAAAAAAP//vx4AAAAAAAAAAAAHqf//QQAAAAAAAAAAAAD//78eAAAAAAAAAAAAVtX//w4AAAAAAAAAAAAA//+/HgAAAAAAAAAAB5L//90KAAAAAAAAAAAAAP//vx4AAAAAAAAAAEjP//9fBwAAAAAAAAAAAAD//78eAAAAAAAACkCu//+qFQAAAAAAAAAAAAAA//+/HgAAAAACV5nR//+zJAAAAAAAAAAAAAAAAP//6bOoqKioquP///+4NQUAAAAAAAAAAAAAAAD///////////////+mHwIAAAAAAAAAAAAAAAAA////////////7GMdDwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/////////////////5kAAAAAAAAAAAAAAAAAAP///fb29vb29vb29vZ6AAAAAAAAAAAAAAAAAAD//9+LgoKCgoKCgoKCCAAAAAAAAAAAAAAAAAAA//+4GQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//uBkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//7gZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//+4GQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//vSUMDAwMDAwMDAgBAAAAAAAAAAAAAAAAAAD///zcycnJycnJycmdDwAAAAAAAAAAAAAAAAAA////////////////0hAAAAAAAAAAAAAAAAAAAP//7ce7u7u7u7u7u48KAAAAAAAAAAAAAAAAAAD//8EwFhYWFhYWFhYDAAAAAAAAAAAAAAAAAAAA//+4GQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//uBkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//7gZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//+4GQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//uBkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//7gZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///CNAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//7bqJgoKCgoKCgoJBAAAAAAAAAAAAAAAAAAD////89vb29vb29vb2kAAAAAAAAAAAAAAAAAAA1////////////////0oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN7//////////////940AAAAAAAAAAAAAAAAAAD/////////////////TgAAAAAAAAAAAAAAAAAA///qtqenp6enp6enpx0AAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//8MpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///DKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//8YvBgYGBgYGBgYBAAAAAAAAAAAAAAAAAAAA///vr46Ojo6Ojo6OPQMAAAAAAAAAAAAAAAAAAP///////////////5AIAAAAAAAAAAAAAAAAAAD////17Ozs7Ozs7Ox9CAAAAAAAAAAAAAAAAAAA///aakpKSkpKSkpKGgIAAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//8MpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///DKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//8MpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///DKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//8MpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///DKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//wykAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYXKcPZ////8M9wLxcCAAAAAAAAAAAAAAAAABFc0//////////////OSwAAAAAAAAAAAAAAABuC6f/807Owo6eyvur///8AAAAAAAAAAAAAABGH8f//qEkYEwAGFidmnu3tAAAAAAAAAAAAAAJG6f//liADAAAAAAAAABVXhQAAAAAAAAAAAAAW0f//lhYAAAAAAAAAAAAAABUAAAAAAAAAAAAAHf//8FsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEv//44DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADE//+KAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////igAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///4oAAAAAAANDXV1dXV1dTQAAAAAAAAAAAAD///+KAAAAAAANxezs7Ozs7NwAAAAAAAAAAAAA5v//igAAAAAAEeb/////////AAAAAAAAAAAAAL3//4oAAAAAAAZWd3d3lOr//wAAAAAAAAAAAABs//+5GAAAAAAABAYGBivO//8AAAAAAAAAAAAAHPr/8GABAAAAAAAAAAAky///AAAAAAAAAAAAABG7//+rJQAAAAAAAAAAJMv//wAAAAAAAAAAAAACRev//6EeAAAAAAAAACTL//8AAAAAAAAAAAAAABCG9f/yqG0TAAAAAydu3///AAAAAAAAAAAAAAAAIaD3///usKOjo6W+4/r/5wAAAAAAAAAAAAAAAAMcWsD/////////////t1sAAAAAAAAAAAAAAAAAAAUUPofi////+alTRBoGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="""

if IS_WINDOWS:
    FONT_CANDIDATES = ["msyh.ttc", "msyh.ttf", "arial.ttf", "arialbd.ttf", "segoeui.ttf"]
else:
    FONT_CANDIDATES = ["LiberationSans-Bold.ttf", "LiberationSans-Regular.ttf",
                       "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]


class StopFlag:
    def __init__(self):
        self._stop = threading.Event()
        self._hook = None

    def start(self, hotkey="esc"):
        keyboard.add_hotkey(hotkey, self._stop.set)
        print(f"[热键] 已注册中断热键 {hotkey.upper()}")

    @property
    def stopped(self):
        return self._stop.is_set()

    def reset(self):
        self._stop.clear()


STOP = StopFlag()


# 屏幕与框选

def select_region():
    if IS_WINDOWS:
        return _select_region_windows()
    return _select_region_x11()


def _select_region_x11():
    import tkinter as tk
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    root.wait_visibility(root)
    try:
        root.attributes("-alpha", 0.3)
    except Exception:
        pass

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    state = {"x0": 0, "y0": 0, "rect": None, "done": False}

    def on_press(event):
        state["x0"], state["y0"] = event.x_root, event.y_root
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x_root, event.y_root, event.x_root, event.y_root,
            outline="red", width=2)

    def on_drag(event):
        if state["rect"]:
            canvas.coords(state["rect"], state["x0"], state["y0"], event.x_root, event.y_root)
            w = abs(event.x_root - state["x0"])
            h = abs(event.y_root - state["y0"])
            canvas.itemconfig(state["size_text"],
                              text=f"框选尺寸: {w} x {h}  (数独应接近正方形且占满区域)")
            canvas.coords(state["size_text"], 10, 45)

    def on_release(event):
        state["done"] = True
        state["x1"], state["y1"] = event.x_root, event.y_root

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: (root.destroy(), sys.exit(0)))
    root.bind("<Return>", lambda e: (setattr(state, "done", True), root.quit()))

    canvas.create_text(10, 10, anchor="nw", fill="white", font=("DejaVu Sans", 16),
                       text="按住鼠标左键框选数独区域, 松开确认 (ESC 取消)")
    state["size_text"] = canvas.create_text(10, 45, anchor="nw", fill="yellow",
                                            font=("DejaVu Sans", 14), text="框选尺寸: -")

    while not state["done"]:
        root.update()
        time.sleep(0.02)
    root.destroy()

    x0, y0 = min(state["x0"], state["x1"]), min(state["y0"], state["y1"])
    x1, y1 = max(state["x0"], state["x1"]), max(state["y0"], state["y1"])
    if x1 - x0 < 10 or y1 - y0 < 10:
        sys.exit("框选区域过小, 退出")
    return (x0, y0), (x1, y1)


def _select_region_windows():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    try:
        tk_w = root.winfo_screenwidth()
        tk_h = root.winfo_screenheight()
        with mss.MSS() as _sct:
            _mon = (_sct.monitors[1] if len(_sct.monitors) > 1 else _sct.monitors[0])
            phys_w = _mon["width"]
            phys_h = _mon["height"]
        sx = (phys_w / tk_w) if tk_w else 1.0
        sy = (phys_h / tk_h) if tk_h else 1.0
    except Exception:
        sx = sy = 1.0
    root.destroy()

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.3)
    root.configure(bg="black")
    root.wait_visibility(root)

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    state = {"x0": 0, "y0": 0, "rect": None, "done": False}

    def on_press(event):
        state["x0"], state["y0"] = event.x_root, event.y_root
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x_root, event.y_root, event.x_root, event.y_root,
            outline="red", width=2)

    def on_drag(event):
        if state["rect"]:
            canvas.coords(state["rect"], state["x0"], state["y0"], event.x_root, event.y_root)
            w = abs(event.x_root - state["x0"])
            h = abs(event.y_root - state["y0"])
            canvas.itemconfig(state["size_text"],
                              text=f"框选尺寸: {w} x {h}  (数独应接近正方形且占满区域)")
            canvas.coords(state["size_text"], 10, 60)

    def on_release(event):
        state["done"] = True
        state["x1"], state["y1"] = event.x_root, event.y_root

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<Return>", lambda e: root.destroy())

    canvas.create_text(10, 10, anchor="nw", fill="white", font=("Microsoft YaHei", 16),
                       text="拖动鼠标框选数独区域\n松开鼠标完成, 按 ESC 取消")
    state["size_text"] = canvas.create_text(10, 60, anchor="nw", fill="yellow",
                                            font=("Microsoft YaHei", 13), text="框选尺寸: -")

    while not state["done"]:
        root.update()
        time.sleep(0.02)
    root.destroy()

    x0, y0 = min(state["x0"], state["x1"]), min(state["y0"], state["y1"])
    x1, y1 = max(state["x0"], state["x1"]), max(state["y0"], state["y1"])
    if x1 - x0 < 10 or y1 - y0 < 10:
        sys.exit("框选区域过小, 退出")

    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        print(f"[DPI] tkinter 坐标为逻辑像素, 换算比例 x={sx:.2f} y={sy:.2f}")
    return (int(round(x0 * sx)), int(round(y0 * sy))), \
        (int(round(x1 * sx)), int(round(y1 * sy)))


def grab_screen(bbox):
    left, top = bbox[0]
    right, bottom = bbox[1]
    monitor = {
        "left": left, "top": top,
        "width": right - left, "height": bottom - top,
    }

    with mss.MSS() as sct:
        shot = sct.grab(monitor)

    img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# 网格定位

def locate_grid(img):
    if img is None or img.size == 0:
        return None, None
    cands = collect_grid_candidates(img)
    if not cands:
        return None, None
    return cands[0][0], cands[0][1]


def collect_grid_candidates(img):
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7)

    cands = []
    seen = set()

    def add(corners, mode):
        key = (int(round(corners[0, 0])), int(round(corners[0, 1])),
               int(round(corners[2, 0])), int(round(corners[2, 1])))
        if key in seen:
            return
        seen.add(key)
        cands.append((corners.astype(np.float32), mode))

    for cand in _collect_contour_candidates(thresh, gray):
        c2 = cand.reshape(4, 2)
        if _looks_like_grid(c2, img):
            add(_order_points(c2), "contour")

    corners, mode = _locate_by_hough(thresh, gray)
    if corners is not None:
        add(corners, "hough")

    whole = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    if _looks_like_grid(whole, img):
        add(whole, "bbox")

    return cands


def _collect_contour_candidates(thresh, gray):
    h, w = thresh.shape[:2]
    min_area = (h * w) * 0.03
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        if bw < 10 or bh < 10:
            continue
        ratio = bw / bh
        if not (0.6 < ratio < 1.7):
            continue
        cands.append((area, approx))
    cands.sort(key=lambda t: -t[0])
    return [c for _, c in cands]


def _looks_like_grid(corners, img):
    try:
        xs = corners[:, 0]
        ys = corners[:, 1]
        if (xs.max() - xs.min()) < 20 or (ys.max() - ys.min()) < 20:
            return False
        grid, _ = extract_grid(img, corners, size=450)
        if grid is None:
            return False
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hh, ww = bw.shape

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, ww // 12), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, hh // 12)))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

        def real_long_lines(lines_img, axis, min_gap=5):
            if axis == 'h':
                raw = [y for y in range(hh)
                       if np.count_nonzero(lines_img[y, :]) > ww * 0.5]
            else:
                raw = [x for x in range(ww)
                       if np.count_nonzero(lines_img[:, x]) > hh * 0.5]
            if not raw:
                return []
            merged = [raw[0]]
            for p in raw[1:]:
                if p - merged[-1] <= min_gap:
                    merged[-1] = p
                else:
                    merged.append(p)
            return merged

        h_real = real_long_lines(h_lines, 'h')
        v_real = real_long_lines(v_lines, 'v')

        if not (5 <= len(h_real) <= 18):
            return False
        if not (5 <= len(v_real) <= 18):
            return False

        def linearity_score(positions):
            if len(positions) < 5:
                return False
            y = np.array(positions, dtype=np.float64)
            x = np.arange(len(y), dtype=np.float64)
            n = len(x)
            sx, sy = x.sum(), y.sum()
            sxx = (x * x).sum()
            sxy = (x * y).sum()
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-9:
                return False
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            y_pred = slope * x + intercept
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot < 1e-9:
                return True
            r2 = 1.0 - ss_res / ss_tot
            return r2 > 0.95

        return linearity_score(h_real) and linearity_score(v_real)
    except Exception:
        return True


def _order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _locate_by_hough(thresh, gray):
    h, w = thresh.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(min(h, w) * 0.3),
                            maxLineGap=15)
    if lines is None or len(lines) < 8:
        return None, None

    horiz, vert = [], []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = math.hypot(dx, dy)
        if length < min(h, w) * 0.2:
            continue
        if dy < dx * 0.3:
            horiz.append((x1, y1, x2, y2))
        elif dx < dy * 0.3:
            vert.append((x1, y1, x2, y2))

    def cluster_lines(lines_list, axis="h"):
        if not lines_list:
            return []
        lines_list = sorted(lines_list, key=lambda l: (l[1] if axis == "h" else l[0]))
        clusters = []
        for l in lines_list:
            pos = (l[1] + l[3]) / 2 if axis == "h" else (l[0] + l[2]) / 2
            if clusters and abs(pos - clusters[-1][0]) < 12:
                total = clusters[-1][1] + 1
                clusters[-1][0] = (clusters[-1][0] * clusters[-1][1] + pos) / total
                clusters[-1][1] = total
            else:
                clusters.append([pos, 1])
        return [c[0] for c in clusters]

    h_clusters = cluster_lines(horiz, "h")
    v_clusters = cluster_lines(vert, "v")

    if len(h_clusters) < 5 or len(v_clusters) < 5:
        return None, None

    def pick_uniform(clusters):
        best, best_score = None, -1.0
        for expected in (17, 15, 13, 11, 9, 7):
            if len(clusters) < expected:
                continue
            for i in range(len(clusters) - expected + 1):
                window = clusters[i:i + expected]
                diffs = np.diff(window)
                if len(diffs) == 0:
                    continue
                std = float(np.std(diffs))
                span = window[-1] - window[0]
                score = span / (std + 1e-6)
                if score > best_score:
                    best_score, best = score, window
        return list(best) if best else clusters

    h_sel = pick_uniform(h_clusters)
    v_sel = pick_uniform(v_clusters)
    if len(h_sel) < 5 or len(v_sel) < 5:
        return None, None

    top = min(h_sel)
    bottom = max(h_sel)
    left = min(v_sel)
    right = max(v_sel)
    if bottom - top < 20 or right - left < 20:
        return None, None

    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                    dtype=np.float32), "hough"


def extract_grid(img, corners, size=720):
    if corners is None:
        return None, None
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M


# 网格线与区域

def find_grid_lines(bw):
    h, w = bw.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 12), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 12)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    def find_full(lines_img, axis):
        if axis == 'h':
            raw = [y for y in range(h) if np.count_nonzero(lines_img[y, :]) > w * 0.5]
        else:
            raw = [x for x in range(w) if np.count_nonzero(lines_img[:, x]) > h * 0.5]
        if not raw:
            return []
        merged = [raw[0]]
        for p in raw[1:]:
            if p - merged[-1] <= 3:
                merged[-1] = p
            else:
                merged.append(p)
        return merged

    return find_full(h_lines, 'h'), find_full(v_lines, 'v')


def infer_n(hy, vx):
    cands = []
    for arr in (hy, vx):
        if len(arr) >= 2:
            cands.append(len(arr) - 1)
            gaps = np.diff(arr)
            med = float(np.median(gaps)) if len(gaps) else 0.0
            if med > 1:
                cands.append(int(round((arr[-1] - arr[0]) / med)))
    if not cands:
        return None
    n = Counter(cands).most_common(1)[0][0]
    if MIN_SIZE <= n <= MAX_SIZE:
        return n
    return None


def regularize_lines(pos, total, n):
    if pos is None or len(pos) < 2:
        return None
    if len(pos) > n + 1:
        best, best_score = None, -1.0
        for i in range(len(pos) - (n + 1) + 1):
            window = pos[i:i + n + 1]
            diffs = np.diff(window)
            if len(diffs) == 0:
                continue
            std = float(np.std(diffs))
            span = window[-1] - window[0]
            score = span / (std + 1e-6)
            if score > best_score:
                best_score, best = score, window
        if best is None:
            return None
        pos = list(best)
    first, last = pos[0], pos[-1]
    step = (last - first) / float(n)
    diffs = np.diff(pos)
    med = float(np.median(diffs)) if len(diffs) else step
    if med < step * 0.85:
        step = med
        last = first + step * n
    if last > total - 1:
        last = pos[-1]
        first = last - step * n
    reg = [round(first + i * step) for i in range(n + 1)]
    return [min(max(0, v), total - 1) for v in reg]


def _edge_thickness_h(bw, y, x0, x1):
    seg = bw[max(0, y - 12):y + 13, x0:x1]
    widths = []
    for col in range(seg.shape[1]):
        dark = np.where(seg[:, col] > 0)[0]
        if len(dark):
            widths.append(dark[-1] - dark[0] + 1)
    return int(np.median(widths)) if widths else 0


def _edge_thickness_v(bw, x, y0, y1):
    seg = bw[y0:y1, max(0, x - 12):x + 13]
    widths = []
    for row in range(seg.shape[0]):
        dark = np.where(seg[row, :] > 0)[0]
        if len(dark):
            widths.append(dark[-1] - dark[0] + 1)
    return int(np.median(widths)) if widths else 0


def collect_edge_thicknesses(bw, hy, vx):
    n = len(hy) - 1
    thicks = []
    for r in range(n - 1):
        y = hy[r + 1]
        for c in range(n):
            x0, x1 = vx[c] + 2, vx[c + 1] - 2
            if x1 <= x0:
                continue
            thicks.append(_edge_thickness_h(bw, y, x0, x1))
    for c in range(n - 1):
        x = vx[c + 1]
        for r in range(n):
            y0, y1 = hy[r] + 2, hy[r + 1] - 2
            if y1 <= y0:
                continue
            thicks.append(_edge_thickness_v(bw, x, y0, y1))
    return np.array(thicks)


def _otsu_threshold(thicks):
    if len(thicks) < 2:
        return 3
    lo, hi = int(thicks.min()), int(thicks.max())
    if hi <= lo:
        return lo + 1
    hist = np.bincount(thicks, minlength=hi + 1)
    total = len(thicks)
    best_t, best_var = None, -1.0
    for t in range(lo, hi + 1):
        w0 = hist[:t + 1].sum()
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = np.dot(np.arange(t + 1), hist[:t + 1]) / w0
        m1 = np.dot(np.arange(t + 1, hi + 1), hist[t + 1:]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t if best_t is not None else 3


def detect_regions(bw, hy, vx, thin_max):
    n = len(hy) - 1
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(n):
        for c in range(n):
            parent[(r, c)] = (r, c)

    for r in range(n - 1):
        y = hy[r + 1]
        for c in range(n):
            x0, x1 = vx[c] + 2, vx[c + 1] - 2
            if x1 <= x0:
                continue
            th = _edge_thickness_h(bw, y, x0, x1)
            if th == 0 or th <= thin_max:
                union((r, c), (r + 1, c))
    for c in range(n - 1):
        x = vx[c + 1]
        for r in range(n):
            y0, y1 = hy[r] + 2, hy[r + 1] - 2
            if y1 <= y0:
                continue
            th = _edge_thickness_v(bw, x, y0, y1)
            if th == 0 or th <= thin_max:
                union((r, c), (r, c + 1))

    groups = defaultdict(list)
    for r in range(n):
        for c in range(n):
            groups[find((r, c))].append((r, c))
    return list(groups.values())


def _valid_regions(regions, n):
    return len(regions) == n and all(len(reg) == n for reg in regions)


def _edge_thickness_maps(bw, hy, vx):
    """全部边厚度算一次(hth[r,c]: 行 r/r+1 间横线; vth[c,r]: 列 c/c+1 间竖线)。

    -1 表示 detect_regions 中"区间退化被 continue"的边(不做合并)。
    旧实现 detect_regions 每个候选 thin_max 都重新逐边测厚, 同一份
    bw/hy/vx 要测 7~8 遍; 边厚与 thin_max 无关, 抽出一次即可。"""
    n = len(hy) - 1
    hth = np.full((max(n - 1, 0), max(n, 0)), -1, dtype=np.int64)
    vth = np.full((max(n - 1, 0), max(n, 0)), -1, dtype=np.int64)
    for r in range(n - 1):
        y = hy[r + 1]
        for c in range(n):
            x0, x1 = vx[c] + 2, vx[c + 1] - 2
            if x1 > x0:
                hth[r, c] = _edge_thickness_h(bw, y, x0, x1)
    for c in range(n - 1):
        x = vx[c + 1]
        for r in range(n):
            y0, y1 = hy[r] + 2, hy[r + 1] - 2
            if y1 > y0:
                vth[c, r] = _edge_thickness_v(bw, x, y0, y1)
    return hth, vth


def _regions_from_maps(hth, vth, n, thin_max):
    """与 detect_regions 完全同构(并查集/合并条件/分组顺序一致), 仅复用预计算边厚。"""
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(n):
        for c in range(n):
            parent[(r, c)] = (r, c)

    for r in range(n - 1):
        for c in range(n):
            th = int(hth[r, c])
            if th >= 0 and (th == 0 or th <= thin_max):
                union((r, c), (r + 1, c))
    for c in range(n - 1):
        for r in range(n):
            th = int(vth[c, r])
            if th >= 0 and (th == 0 or th <= thin_max):
                union((r, c), (r, c + 1))

    groups = defaultdict(list)
    for r in range(n):
        for c in range(n):
            groups[find((r, c))].append((r, c))
    return list(groups.values())


def best_thin_max(bw, hy, vx):
    n = len(hy) - 1
    thicks = collect_edge_thicknesses(bw, hy, vx)
    if len(thicks) == 0:
        return 3, detect_regions(bw, hy, vx, 3)
    hth, vth = _edge_thickness_maps(bw, hy, vx)
    med = float(np.median(thicks))
    otsu = _otsu_threshold(thicks)
    cands = sorted(set([
        int(round(med * 1.25)), int(round(med * 1.5)), int(round(med * 1.75)),
        int(med) + 1, int(med) + 2, otsu,
    ]))
    for t in cands:
        if t <= 0:
            continue
        regions = _regions_from_maps(hth, vth, n, t)
        if _valid_regions(regions, n):
            return t, regions
    return otsu, _regions_from_maps(hth, vth, n, otsu)


def detect_box_struct(grid_img):
    """标准数独矩形宫格回退: 检测宫格尺寸 (box_h, box_w), 失败返回 None."""
    try:
        gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = bw.shape

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 30), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 30)))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

        def find_lines(lines_img, axis):
            if axis == 'h':
                raw = [y for y in range(h) if np.count_nonzero(lines_img[y, :]) > w * 0.5]
            else:
                raw = [x for x in range(w) if np.count_nonzero(lines_img[:, x]) > h * 0.5]
            if not raw:
                return []
            merged = [raw[0]]
            for p in raw[1:]:
                if p - merged[-1] <= 3:
                    merged[-1] = p
                else:
                    merged.append(p)
            return merged

        ys = find_lines(h_lines, 'h')
        xs = find_lines(v_lines, 'v')
        if len(ys) < 5 or len(xs) < 5:
            return None, None

        def infer_n_from_lines(pos_lines, total_span, candidates):
            if pos_lines is None or len(pos_lines) < 5:
                return None
            first, last = pos_lines[0], pos_lines[-1]
            span = last - first
            if span < total_span * 0.3:
                return None
            best_n, best_score = None, -1.0
            for n in candidates:
                step = span / float(n)
                tol = max(step * 0.35, 2.0)
                expected = [first + i * step for i in range(n + 1)]
                matched = sum(1 for p in pos_lines
                              if any(abs(p - e) < tol for e in expected))
                missing = sum(1 for e in expected
                              if not any(abs(p - e) < tol for p in pos_lines))
                coverage = matched / len(pos_lines)
                score = coverage - 0.25 * (missing / (n + 1))
                if score > best_score:
                    best_score, best_n = score, n
            return best_n if best_score >= 0.5 else None

        n_h = infer_n_from_lines(ys, h, (9, 12, 16))
        n_v = infer_n_from_lines(xs, w, (9, 12, 16))
        n = n_h if n_h is not None else n_v
        if n is None:
            return None, None

        def line_thicknesses(positions, axis):
            out = []
            for p in positions:
                if axis == 'h':
                    seg = bw[max(0, p - 6):p + 7, :].mean(axis=1)
                else:
                    seg = bw[:, max(0, p - 6):p + 7].mean(axis=0)
                dark = np.where(seg > 0.5)[0]
                out.append(dark[-1] - dark[0] + 1 if len(dark) else 1)
            return out

        def detect_box_dim(thicks, n, prefer):
            med = float(np.median(thicks))
            coarse = [i for i, t in enumerate(thicks) if t >= med * 1.5 + 0.5]
            if len(coarse) >= 3:
                gaps = np.diff(coarse)
                g = int(Counter(gaps).most_common(1)[0][0])
                if 2 <= g <= (n + 1) // 2 and (n % g) == 0:
                    return g
            return prefer

        std_boxes = {9: (3, 3), 12: (3, 4), 16: (4, 4)}
        box_h, box_w = std_boxes[n]
        yt = line_thicknesses(ys, 'h')
        xt = line_thicknesses(xs, 'v')
        bh = detect_box_dim(yt, n, box_h)
        bw_ = detect_box_dim(xt, n, box_w)
        return bh, bw_
    except Exception:
        return None, None


def box_regions(n, box_h, box_w):
    regions = []
    for bi in range(0, n, box_h):
        for bj in range(0, n, box_w):
            regions.append([(r, c) for r in range(bi, bi + box_h)
                            for c in range(bj, bj + box_w)])
    return regions


def split_cells_irregular(grid_img, hy, vx):
    n = len(hy) - 1
    cells = []
    ok = True
    for r in range(n):
        y0, y1 = hy[r], hy[r + 1]
        if y1 - y0 < 3:
            ok = False
            break
        for c in range(n):
            x0, x1 = vx[c], vx[c + 1]
            if x1 - x0 < 3:
                ok = False
                break
            margin = 3
            cell = grid_img[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
            if cell.size == 0:
                ok = False
                break
            cells.append(cell)
        if not ok:
            break
    if ok and len(cells) == n * n:
        cw = vx[1] - vx[0] if vx[1] > vx[0] else (vx[-1] - vx[0]) // n
        return cells, max(cw, 1)
    return [], 0


# 符号识别

def decode_embedded_templates():
    import base64 as _b64
    import struct as _st

    tpls = {}
    if EMBEDDED_TEMPLATE_B64:
        try:
            raw = _b64.b64decode(EMBEDDED_TEMPLATE_B64)
            size = _st.unpack('<I', raw[:4])[0]
            nbytes = size * size
            for idx, ch in enumerate(SYMBOLS):
                off = 4 + idx * nbytes
                arr = np.frombuffer(raw[off:off + nbytes], dtype=np.uint8).reshape(size, size)
                if arr.max() > 0:
                    tpls[ch] = arr.astype(np.float32) / 255.0
        except Exception:
            tpls = {}

    sys_tpls = _generate_system_templates()
    for ch in SYMBOLS:
        if ch not in tpls and ch in sys_tpls:
            tpls[ch] = sys_tpls[ch]
    return tpls


def _find_font():
    if IS_WINDOWS:
        windir = os.environ.get("WINDIR", "C:\\Windows")
        for fname in FONT_CANDIDATES:
            p = os.path.join(windir, "Fonts", fname)
            if os.path.exists(p):
                return p
        return None
    for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNSText.ttf"]:
        if os.path.exists(p):
            return p
    return None


def _generate_system_templates():
    font_path = _find_font()
    if font_path is None:
        return {}
    templates = {}
    canvas = 64
    for ch in SYMBOLS:
        img = Image.new("L", (canvas, canvas), 0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, int(canvas * 0.75))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((canvas - tw) / 2 - bbox[0], (canvas - th) / 2 - bbox[1]),
                  ch, fill=255, font=font)
        arr = np.array(img, dtype=np.uint8)
        # 与内嵌模板(实测内容最大维度恒为 22)和 _cell_to_template(size-6)
        # 三方统一; 旧值 -4(内容24)与运行时规则不一致, 每次匹配先天掉分。
        scale = (TEMPLATE_SIZE - 6) / max(arr.shape)
        new_w = max(1, int(arr.shape[1] * scale))
        new_h = max(1, int(arr.shape[0] * scale))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas_img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
        y0 = (TEMPLATE_SIZE - new_h) // 2
        x0 = (TEMPLATE_SIZE - new_w) // 2
        canvas_img[y0:y0 + new_h, x0:x0 + new_w] = arr
        _, canvas_img = cv2.threshold(canvas_img, 100, 255, cv2.THRESH_BINARY)
        templates[ch] = canvas_img.astype(np.float32) / 255.0
    return templates


def _cell_to_template(cell_bgr, size=28):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape

    row_ink = (bw > 0).sum(axis=1) / w
    col_ink = (bw > 0).sum(axis=0) / h
    for y in range(h):
        if row_ink[y] > 0.8:
            bw[y, :] = 0
    for x in range(w):
        if col_ink[x] > 0.8:
            bw[:, x] = 0

    ncc, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    for i in range(1, ncc):
        x, y, cw, ch, area = stats[i]
        touches = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
        if not touches and area >= 4:
            comps.append((int(area), int(x), int(y), int(cw), int(ch)))
    if not comps:
        return None

    max_area = max(c[0] for c in comps)
    keep = [c for c in comps if c[0] >= max_area * 0.15]
    xs = [c[1] for c in keep]
    ys = [c[2] for c in keep]
    x2s = [c[1] + c[3] for c in keep]
    y2s = [c[2] + c[4] for c in keep]
    x, y = min(xs), min(ys)
    cw, ch = max(x2s) - x, max(y2s) - y
    if cw < 3 or ch < 3:
        return None
    digit = bw[y:y + ch, x:x + cw]
    if digit.max() == 0:
        return None
    scale = (size - 6) / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas_img = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas_img[y0:y0 + new_h, x0:x0 + new_w] = digit
    return canvas_img / 255.0


def _cell_to_template_rel(cell_bgr, size=28):
    """极性救援版模板提取: 格内区(中心 64%) + 对格体众数的偏离图 Otsu。

    主路径的"墨迹=暗像素"隐含深字浅底极性; 浅灰数字/暗底亮字会整盘读空,
    而读空不是安全失败——solve 会给空盘解出合法解并全盘填错(实测)。
    本函数: 先裁内区把网格线(最亮/最暗的干扰源)挡在外面, 再以"偏离格体
    众数"统一两种极性的笔画, Otsu 只需在 背景/字 两类间切。偏离 max<25
    或前景占比>45% 判无数字; 组件按面积(>=15%最大块)保留、不做贴边剔除
    (内区边界不是格线)。仅在"已知格低于护栏下限"时被调用, 常规盘不走。
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0, y1 = int(round(h * 0.18)), int(round(h * 0.82))
    x0, x1 = int(round(w * 0.18)), int(round(w * 0.82))
    inner = gray[y0:y1, x0:x1]
    if inner.size < 25 or min(inner.shape) < 6:
        return None
    body = int(np.argmax(np.bincount(inner.ravel())))
    diff = np.abs(inner.astype(np.int16) - body).astype(np.uint8)
    if int(diff.max()) < 25:
        return None
    _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((bw > 0).mean()) > 0.45:
        return None

    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if ncc <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    keep = np.where((areas >= max_area * 0.15) & (areas >= 4))[0] + 1
    if len(keep) == 0:
        return None
    sel = np.isin(labels, keep).astype(np.uint8) * 255
    ys, xs = np.where(sel > 0)
    y_, x_ = ys.min(), xs.min()
    ch, cw = ys.max() - y_ + 1, xs.max() - x_ + 1
    if cw < 3 or ch < 3:
        return None
    digit = sel[y_:y_ + ch, x_:x_ + cw]
    scale = (size - 6) / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas_img = np.zeros((size, size), dtype=np.uint8)
    ty = (size - new_h) // 2
    tx = (size - new_w) // 2
    canvas_img[ty:ty + new_h, tx:tx + new_w] = digit
    return canvas_img / 255.0


def recognize_cells_rel(cells, templates, n=9):
    """极性救援识别: 与 recognize_cells 同判分/同置信闸, 仅模板提取不同。"""
    board = np.zeros((n, n), dtype=int)
    for idx, cell in enumerate(cells):
        if cell is None or idx >= n * n:
            continue
        r, c = divmod(idx, n)
        t = _cell_to_template_rel(cell)
        if t is None:
            continue
        t = t.astype(np.float32)
        best_i, best_v = 0, -1.0
        for i2, ch in enumerate(SYMBOLS[:n], start=1):
            if ch not in templates:
                continue
            res = cv2.matchTemplate(t, templates[ch], cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            if mv > best_v:
                best_v, best_i = mv, i2
        if best_v >= 0.5:      # 与主路径 _recognize_cell_scored 同一置信闸
            board[r, c] = best_i
    return board


def _recognize_cell_scored(cell, templates, n=None):
    if cell is None or cell.size == 0:
        return 0, 0.0, 0.0
    if n is None:
        n = 9
    symbols = SYMBOLS[:n]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    center = gray[cy0:cy1, cx0:cx1]
    if center.size == 0:
        return 0, 0.0, 0.0
    ink = np.count_nonzero(center < 200) / center.size
    if ink < 0.02:
        return 0, ink, 0.0
    t = _cell_to_template(cell)
    if t is None:
        return 0, ink, 0.0
    t = t.astype(np.float32)
    best_i, best_v = 0, -1.0
    for idx, ch in enumerate(symbols, start=1):
        if ch not in templates:
            continue
        res = cv2.matchTemplate(t, templates[ch], cv2.TM_CCOEFF_NORMED)
        _, mv, _, _ = cv2.minMaxLoc(res)
        if mv > best_v:
            best_v, best_i = mv, idx
    if best_v < 0.5:
        return 0, ink, best_v
    return best_i, ink, best_v


def recognize_cell(cell_bgr, templates, n=9):
    if cell_bgr is None or cell_bgr.size == 0:
        return 0
    d, ink, conf = _recognize_cell_scored(cell_bgr, templates, n)
    return d


def recognize_cells(cells, templates, n=9):
    board = np.zeros((n, n), dtype=int)
    for idx, cell in enumerate(cells):
        if cell is None or idx >= n * n:
            continue
        r, c = divmod(idx, n)
        board[r, c] = recognize_cell(cell, templates, n)
    return board


def _bootstrap_recognize(cells, base_tpls, n):
    symbols = SYMBOLS[:n]
    collect = defaultdict(list)
    for idx, cell in enumerate(cells):
        if idx >= n * n:
            break
        d, ink, conf = _recognize_cell_scored(cell, base_tpls, n)
        if d != 0 and conf >= 0.6 and d <= len(symbols):
            t = _cell_to_template(cell)
            if t is not None:
                collect[d].append(t)

    usable = {d: v for d, v in collect.items() if len(v) >= 2}
    if len(usable) < 5:
        return None

    refined = {}
    for d, v in usable.items():
        ch = symbols[d - 1]
        refined[ch] = np.mean(v, axis=0).astype(np.float32)
    for ch in symbols:
        if ch not in refined and ch in base_tpls:
            refined[ch] = base_tpls[ch]

    board2 = np.zeros((n, n), dtype=int)
    for idx, cell in enumerate(cells):
        if idx >= n * n:
            break
        r, c = divmod(idx, n)
        board2[r, c] = recognize_cell(cell, refined, n)
    return board2


# 求解

def solve_sudoku(board, regions):
    """通用求解器: 每行唯一 + 每列唯一 + 每个区域(宫格/不规则)唯一.

    regions 为一组格子列表, 每个列表代表一个区域. 标准数独的矩形宫格与
    异形数独的不规则区域在此统一, 无需区分题型.
    """
    if board is None or regions is None or len(regions) == 0:
        return None
    n = board.shape[0]
    cell_region = {}
    for rid, reg in enumerate(regions):
        for (r, c) in reg:
            cell_region[(r, c)] = rid

    units = []
    for r in range(n):
        units.append([(r, c) for c in range(n)])
    for c in range(n):
        units.append([(r, c) for r in range(n)])
    for reg in regions:
        units.append(list(reg))

    cell_units = {}
    for (r, c) in cell_region:
        cell_units[(r, c)] = [u for u in units if (r, c) in u]

    def is_complete(cur):
        for u in units:
            seen = set()
            for (r, c) in u:
                v = cur[r, c]
                if v == 0:
                    return False
                if v in seen:
                    return False
                seen.add(v)
        return True

    def candidates(cur, r, c):
        used = set()
        for u in cell_units[(r, c)]:
            for (rr, cc) in u:
                used.add(cur[rr, cc])
        used.discard(0)
        return [v for v in range(1, n + 1) if v not in used]

    def propagate(cur):
        while True:
            if STOP.stopped:
                return False
            progressed = False
            for r in range(n):
                for c in range(n):
                    if cur[r, c] == 0:
                        cs = candidates(cur, r, c)
                        if not cs:
                            return False
                        if len(cs) == 1:
                            cur[r, c] = cs[0]
                            progressed = True
            for u in units:
                pos = {}
                for (r, c) in u:
                    if cur[r, c] == 0:
                        for v in candidates(cur, r, c):
                            pos.setdefault(v, []).append((r, c))
                for v, cells in pos.items():
                    if len(cells) == 1:
                        rr, cc = cells[0]
                        if cur[rr, cc] == 0:
                            cur[rr, cc] = v
                            progressed = True
            if not progressed:
                break
        return True

    def dfs(cur):
        if STOP.stopped:
            return None
        cur = cur.copy()
        if not propagate(cur):
            return None
        done = True
        best = None
        for r in range(n):
            for c in range(n):
                if cur[r, c] == 0:
                    done = False
                    cs = candidates(cur, r, c)
                    if not cs:
                        return None
                    if best is None or len(cs) < best[0]:
                        best = (len(cs), r, c, cs)
        if done:
            return cur.copy() if is_complete(cur) else None
        _, r, c, cs = best
        for v in cs:
            cur[r, c] = v
            res = dfs(cur)
            if res is not None:
                return res
            cur[r, c] = 0
        return None

    return dfs(board)


def _find_conflicts(board, regions):
    n = board.shape[0]
    conflicted = set()
    for i in range(n):
        row = [j for j in range(n) if board[i, j]]
        for j1 in range(len(row)):
            for j2 in range(j1 + 1, len(row)):
                if board[i, row[j1]] == board[i, row[j2]]:
                    conflicted.add((i, row[j1]))
                    conflicted.add((i, row[j2]))
    for j in range(n):
        col = [i for i in range(n) if board[i, j]]
        for i1 in range(len(col)):
            for i2 in range(i1 + 1, len(col)):
                if board[col[i1], j] == board[col[i2], j]:
                    conflicted.add((col[i1], j))
                    conflicted.add((col[i2], j))
    for reg in regions:
        cells_box = [(r, c) for (r, c) in reg if board[r, c]]
        for c1 in range(len(cells_box)):
            for c2 in range(c1 + 1, len(cells_box)):
                if board[cells_box[c1]] == board[cells_box[c2]]:
                    conflicted.add(cells_box[c1])
                    conflicted.add(cells_box[c2])
    return conflicted


def _auto_repair(board, regions, cells=None, tpls=None):
    n = board.shape[0]
    if solve_sudoku(board, regions) is not None:
        return board

    confusable = {
        'B': ['8', 'E', 'D'], '8': ['B', '3', '6'], '6': ['5', '8', '9'],
        '5': ['6', '3'], '3': ['8', '5'], '9': ['6', '7'], 'D': ['B', 'O', '0'],
        'A': ['4'], '4': ['A', '9'], '1': ['7'], '7': ['1', '9'], 'E': ['F', 'B'],
        'F': ['E', 'P'], 'C': ['G', 'O'], 'G': ['C', '6'], '2': ['Z', '7'],
    }

    cand = {}
    for i in range(n):
        for j in range(n):
            if board[i, j] == 0:
                continue
            orig_val = board[i, j]
            orig_ch = SYMBOLS[orig_val - 1] if 1 <= orig_val <= len(SYMBOLS) else None
            vals = {orig_val}
            if orig_ch and orig_ch in confusable:
                for ch in confusable[orig_ch]:
                    if ch not in SYMBOLS:
                        continue
                    idx = SYMBOLS.index(ch) + 1
                    if 1 <= idx <= n:
                        vals.add(idx)
            if cells is not None and tpls is not None:
                idx = i * n + j
                cell = cells[idx]
                t = _cell_to_template(cell)
                if t is not None:
                    scores = []
                    for ch in SYMBOLS[:n]:
                        if ch not in tpls:
                            continue
                        res = cv2.matchTemplate(t.astype(np.float32),
                                                tpls[ch], cv2.TM_CCOEFF_NORMED)
                        _, mv, _, _ = cv2.minMaxLoc(res)
                        scores.append((mv, ch))
                    scores.sort(reverse=True)
                    for _, ch in scores[:3]:
                        v = SYMBOLS.index(ch) + 1
                        if 1 <= v <= n:
                            vals.add(v)
            cand[(i, j)] = list(vals)

    fixable = [(i, j) for (i, j), v in cand.items() if len(v) > 1]
    if not fixable:
        return None

    conflicted = _find_conflicts(board, regions)
    search_cells = [c for c in fixable if c in conflicted] if conflicted else fixable
    if not search_cells:
        search_cells = fixable
    if len(search_cells) > 12:
        search_cells = search_cells[:12]

    import itertools
    max_k = 2 if n <= 12 else 1
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(search_cells, k):
            if STOP.stopped:
                return None
            val_options = []
            for (i, j) in combo:
                alts = [v for v in cand[(i, j)] if v != int(board[i, j])]
                if not alts:
                    break
                val_options.append(alts)
            if len(val_options) != k:
                continue
            for vals in itertools.product(*val_options):
                if STOP.stopped:
                    return None
                b2 = board.copy()
                for idx_c, (i, j) in enumerate(combo):
                    b2[i, j] = vals[idx_c]
                if _find_conflicts(b2, regions):
                    continue
                sol = solve_sudoku(b2, regions)
                if sol is not None:
                    return b2
    return None


def _load_templates():
    tpls = decode_embedded_templates()
    if len(tpls) < 9:
        print("[警告] 模板不完整, 符号识别准确率可能下降")
    return tpls


# 自动填入

def compute_cell_centers(origin, grid_offset, grid_size, n=9):
    gx, gy = grid_offset
    gw, gh = grid_size
    cw, ch = gw / float(n), gh / float(n)
    ox, oy = origin
    centers = np.zeros((n, n, 2), dtype=float)
    for i in range(n):
        for j in range(n):
            cx = ox + gx + j * cw + cw / 2.0
            cy = oy + gy + i * ch + ch / 2.0
            centers[i, j] = (cx, cy)
    return centers


def _win_click_press_env():
    """Windows ctypes 快速点击+键入(SetCursorPos+SendInput, <0.5ms/次,
    对比 pyautogui 每次 10~20ms 封装), 定时器提到 1ms 精度.
    press() 数字/字母按 VK 直发, 其余回退 pyautogui.press.
    返回 (click, press, cleanup).
    """
    import ctypes

    ULONG = ctypes.c_ulong
    USHORT = ctypes.c_ushort

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ULONG), ("dwFlags", ULONG),
                    ("time", ULONG), ("dwExtraInfo", ctypes.c_void_p)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", USHORT), ("wScan", USHORT), ("dwFlags", ULONG),
                    ("time", ULONG), ("dwExtraInfo", ctypes.c_void_p)]

    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ULONG), ("u", _U)]

    user32 = ctypes.windll.user32
    winmm = ctypes.windll.winmm
    timer_set = False
    try:
        timer_set = (winmm.timeBeginPeriod(1) == 0)
    except Exception:
        pass

    minputs = (INPUT * 2)()
    minputs[0].type = 0
    minputs[0].u.mi = MOUSEINPUT(0, 0, 0, 0x0002, 0, None)     # LEFTDOWN
    minputs[1].type = 0
    minputs[1].u.mi = MOUSEINPUT(0, 0, 0, 0x0004, 0, None)     # LEFTUP

    def click(x, y):
        if not user32.SetCursorPos(int(x), int(y)):
            raise RuntimeError(f"鼠标定位失败: ({x},{y})")
        if user32.SendInput(2, minputs, ctypes.sizeof(INPUT)) != 2:
            raise RuntimeError(f"鼠标事件发送失败: ({x},{y})")

    _VK = {str(d): 0x30 + d for d in range(10)}
    _VK.update({chr(ord('A') + i): 0x41 + i for i in range(26)})
    _VK.update({chr(ord('a') + i): 0x41 + i for i in range(26)})
    _vk_cache = {}

    def _key_pair(vk):
        pair = _vk_cache.get(vk)
        if pair is None:
            pair = (INPUT * 2)()
            pair[0].type = 1                                   # INPUT_KEYBOARD
            pair[0].u.ki = KEYBDINPUT(vk, 0, 0, 0, None)       # KEYDOWN
            pair[1].type = 1
            pair[1].u.ki = KEYBDINPUT(vk, 0, 0x0002, 0, None)  # KEYUP
            _vk_cache[vk] = pair
        return pair

    def press(ch):
        vk = _VK.get(ch) if len(ch) == 1 else None
        if vk is None:                      # 多字符/特殊符号: 回退等价路径
            pyautogui.press(ch)
            return
        pair = _key_pair(vk)
        if user32.SendInput(2, pair, ctypes.sizeof(INPUT)) != 2:
            raise RuntimeError(f"按键发送失败: {ch}")

    def cleanup():
        if timer_set:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass

    return click, press, cleanup


def _pyautogui_click_press_env():
    """pyautogui 点击+键入环境(非 Windows / 快速路径不可用时的回退)."""
    pyautogui.PAUSE = 0.0
    pyautogui.MINIMUM_DURATION = 0.0
    pyautogui.MINIMUM_SLEEP = 0.0

    def click(x, y):
        pyautogui.click(int(x), int(y))

    def press(ch):
        pyautogui.press(ch)

    def cleanup():
        pass

    return click, press, cleanup


def autofill_board(board, solution, centers):
    """按空格序列点击+键入; Windows 走快速路径, 其余回退 pyautogui."""
    n = board.shape[0]
    filled = 0
    empty_cells = [(r, c) for r in range(n) for c in range(n) if board[r, c] == 0]

    if IS_WINDOWS:
        try:
            click, press, cleanup = _win_click_press_env()
        except Exception:
            click, press, cleanup = _pyautogui_click_press_env()
    else:
        click, press, cleanup = _pyautogui_click_press_env()

    try:
        for r, c in empty_cells:
            if STOP.stopped:
                print(f"[中断] 已填入 {filled} 格, 剩余 {len(empty_cells) - filled} 格未填")
                break
            cx, cy = centers[r, c]

            jx = random.uniform(-CONFIG["jitter"], CONFIG["jitter"])
            jy = random.uniform(-CONFIG["jitter"], CONFIG["jitter"])
            target_x, target_y = int(round(cx + jx)), int(round(cy + jy))

            # click 带坐标会瞬间定位(不播放平滑移动动画)并点击
            click(target_x, target_y)
            time.sleep(CONFIG["click_interval"])

            val = solution[r, c]
            ch = SYMBOLS[val - 1] if 1 <= val <= len(SYMBOLS) else str(val)
            press(ch)   # 单键直发, 比 typewrite 逐字符快
            filled += 1

            time.sleep(CONFIG["cell_delay"])
    finally:
        cleanup()

    return filled


# 主流程

def process_candidate(shot_img, corners, tpls):
    """处理一个候选网格: 矫正 -> 找线 -> 推断尺寸 -> 区域检测 -> 识别.

    区域检测对标准数独(矩形宫格)与异形数独(不规则区域)统一: 线宽法优先,
    失败时回退到矩形宫格检测.
    """
    grid_img, M = extract_grid(shot_img, corners, size=720)
    if grid_img is None:
        return None
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    _, bwth = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = (bwth > 128).astype(np.uint8)

    hy, vx = find_grid_lines(bw)
    n = infer_n(hy, vx)
    if n is None:
        return None
    hyr = regularize_lines(hy, 720, n)
    vxr = regularize_lines(vx, 720, n)
    if hyr is None or vxr is None:
        return None

    thin_max, regions = best_thin_max(bw, hyr, vxr)
    kind = "不规则"
    if not _valid_regions(regions, n):
        box_h, box_w = detect_box_struct(grid_img)
        if box_h is None or box_w is None:
            return None
        # 矩形宫格检测的 n 可能与线数推断不同(视频帧线检测有噪声),
        # 此时以宫格结构为准重新推断 n 并规整网格线
        std_n = n
        for cand_n in (16, 12, 9, n):
            if cand_n % box_h == 0 and cand_n % box_w == 0:
                std_n = cand_n
                break
        hyr = regularize_lines(hy, 720, std_n)
        vxr = regularize_lines(vx, 720, std_n)
        if hyr is None or vxr is None:
            return None
        regions = box_regions(std_n, box_h, box_w)
        n = std_n
        kind = "宫格"

    cells, cell_px = split_cells_irregular(grid_img, hyr, vxr)
    if len(cells) != n * n:
        return None

    board = recognize_cells(cells, tpls, n)
    n_known = int((board != 0).sum())
    return {
        "n": n, "hy": hyr, "vx": vxr, "regions": regions,
        "cells": cells, "board": board, "n_known": n_known,
        "grid_img": grid_img, "thin_max": thin_max, "kind": kind,
    }


def run_pipeline(origin, shot_img):
    candidates = collect_grid_candidates(shot_img)
    if not candidates:
        print("[错误] 无法定位数独网格")
        print("[提示] 请确保框选区域完整包含数独网格,")
        print("       并尽量让数独占据框选区域的绝大部分空间.")
        print("       框选时避免包含 IDE/浏览器等其他界面元素.")
        return None, None, None

    tpls = _load_templates()
    results = []
    for corners, mode in candidates:
        res = process_candidate(shot_img, corners, tpls)
        if res is None:
            continue
        score = res["n_known"] / float(res["n"] * res["n"])
        results.append((score, corners, mode, res))

    if not results:
        print("[错误] 无法定位数独网格")
        print("[提示] 请确保框选区域完整包含数独网格,")
        print("       并尽量让数独占据框选区域的绝大部分空间.")
        print("       框选时避免包含 IDE/浏览器等其他界面元素.")
        return None, None, None

    # 优先尝试多数候选一致的尺寸 (modal n), 再按已知格占比排序
    n_counts = Counter(r[3]["n"] for r in results)
    modal_n = n_counts.most_common(1)[0][0] if n_counts else None
    results.sort(key=lambda r: (r[3]["n"] == modal_n, r[0]), reverse=True)

    for score, corners, mode, res in results:
        if STOP.stopped:
            print("[中断] 识别求解阶段已被停止")
            return None, None, None
        n = res["n"]
        board = res["board"]
        regions = res["regions"]
        cells = res["cells"]

        print(f"[网格] 尺寸: {n}x{n}, 区域数: {len(regions)}, 类型: {res['kind']}, "
              f"定位方式: {mode}, 识别出 {res['n_known']} 个已知格")

        print("[识别] 题目矩阵:")
        for row in board:
            print("  " + " ".join(str(x) for x in row))

        solution = None
        if res["n_known"] >= _min_known(n):
            solution = solve_sudoku(board, regions)
        else:
            print(f"[护栏] 已知格 {res['n_known']} 低于下限 {_min_known(n)}, "
                  "不做盲解, 转入精修/救援链")

        if solution is None:
            print("[提示] 首轮识别可能不准, 尝试自举精修...")
            refined = _bootstrap_recognize(cells, tpls, n)
            if refined is not None:
                board = refined
                print("[识别-精修] 题目矩阵:")
                for row in board:
                    print("  " + " ".join(str(x) for x in row))
                solution = solve_sudoku(board, regions)

        if solution is None:
            print("[提示] 尝试自动修复近似符号混淆...")
            repaired = _auto_repair(board, regions, cells, tpls)
            if repaired is not None:
                print("[修复] 修正后的题目矩阵:")
                for row in repaired:
                    print("  " + " ".join(str(x) for x in row))
                solution = solve_sudoku(repaired, regions)
                if solution is not None:
                    board = repaired

        if solution is None and res["n_known"] < _min_known(n):
            print("[救援] 已知格过少, 尝试少数类极性重读(浅灰字/暗底亮字)...")
            try:
                board_rel = recognize_cells_rel(cells, tpls, n)
            except Exception:
                board_rel = None
            if board_rel is not None and int((board_rel > 0).sum()) > res["n_known"]:
                cand_sol = solve_sudoku(board_rel, regions)
                if cand_sol is not None:
                    print(f"[救援] 极性重读得到 {int((board_rel > 0).sum())} "
                          "个已知格并成功求解, 采用救援结果")
                    board, solution = board_rel, cand_sol

        if solution is not None:
            known_final = int((np.asarray(board) > 0).sum())
            if known_final < _min_known(n):
                print(f"[护栏] 题面已知格仅 {known_final} 个(下限 {_min_known(n)}), "
                      "疑似整盘未被读出; 拒绝作答, 不回填(请重新框选/检查字体)")
                solution = None

        if solution is not None:
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            gx, gy = int(round(x_coords.min())), int(round(y_coords.min()))
            gw, gh = int(round(x_coords.max() - x_coords.min())), int(round(y_coords.max() - y_coords.min()))
            grid_meta = {
                "corners": corners,
                "offset": (gx, gy),
                "size": (gw, gh),
                "mode": mode,
                "n": n,
            }
            return board, solution, grid_meta

        print("[错误] 该候选无解, 尝试下一个候选...")

    print("[错误] 所有候选均无解, 可能是识别错误")
    print("[提示] 请重新框选数独区域再试; 若网页字体与内置模板不一致, 识别可能不准")
    return None, None, None


def _min_known(n):
    """可作答的最少已知格下限: 低于此数几乎必是"整盘没读出来"。
    数独唯一解最少 17 提示(9x9), 各尺寸按 0.9n 取下限是极保守的安全网;
    真正的灾难路径是 n_known≈0 时 solve 仍会返回一个合法解并被全盘
    填入(实测), 这里必须拦住。"""
    return max(5, int(n * 0.9))


def _boost_timer():
    """把 Windows 系统定时器精度提到 1ms(退出时复原)。默认 15.6ms 粒度下
    time.sleep(0.015) 实际会睡到 15.6~31ms, 逐格点击的毫秒级间隔形同虚设。
    姊妹项目(马赛克/扫雷)实测结论。非 Windows 或失败时静默跳过。"""
    import sys
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        if ctypes.windll.winmm.timeBeginPeriod(1) == 0:
            import atexit
            atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)
    except Exception:
        pass


def main():
    _boost_timer()
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    print("=" * 46)
    print("  数独自动识别与回填系统 (标准/异形通用)")
    print("  支持 4x4~16x16 各尺寸, 自动识别宫格或异形区域")
    print("  1. 全屏出现遮罩后, 拖动鼠标框选数独区域")
    print("  2. 确认后自动识别类型并求解")
    print(f"  3. 自动填入, 按 {CONFIG['stop_hotkey'].upper()} 随时中断")
    print("=" * 46)

    origin, bottom = select_region()
    print(f"[框选] 区域: {origin} -> {bottom}")

    STOP.start(CONFIG["stop_hotkey"])
    STOP.reset()

    shot_img = grab_screen((origin, bottom))
    print(f"[截屏] 获取 {shot_img.shape[1]}x{shot_img.shape[0]} 图像")

    board, solution, meta = run_pipeline(origin, shot_img)
    if solution is None or meta is None:
        sys.exit(1)

    n = meta.get("n", 9)
    centers = compute_cell_centers(origin, meta["offset"], meta["size"], n)

    blanks = int((board == 0).sum())
    print(f"\n[确认] 共 {blanks} 个空白格待填, 开始自动填入...")

    filled = autofill_board(board, solution, centers)
    print(f"\n[完成] 成功填入 {filled} 个格子")


if __name__ == "__main__":
    main()
