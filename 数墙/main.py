# -*- coding: utf-8 -*-
"""main.py — 数墙(Nurikabe)自动识别/求解/作答 入口与配置.

依赖: opencv-python numpy Pillow mss pyautogui keyboard ortools
      (ortools 缺失时自动退化为纯自研求解器)

用法:
    python main.py            # 框选→识别→求解→瞬移点击作答→回车提交

配置全部写死在本文件 CFG 中, 路径相对本文件所在目录解析, 无硬编码盘符。
流程中的异常(识别失败/求解超时/无解/用户中断)都会给出明确提示并允许
重试, 不会静默崩溃; 全程 ESC 可中断作答.
"""
import os
import sys
import time

import automation
import recognizer
import sat_engine
import solver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置(全部写死)
CFG = {
    # 路径(相对 BASE_DIR)
    "template_cache": os.path.join(BASE_DIR, "templates.npz"),

    # 图像识别
    "dark_threshold": 128,           # 灰度暗像素阈值(墙体/数字)
    "morph_close_size": 3,           # 墙体闭运算核(失败自动 3/5/7 递增重试)
    "wall_min_frac": 0.01,           # 主墙体组件最小面积占整图比例
    "cell_area_lo": 0.15,            # 单元格内部组件面积下限(×中位数)
    "cell_area_hi": 4.0,             # 单元格内部组件面积上限(×中位数)
    "min_cell_px": 12,               # 单元格最小边长, 过小判定非题目
    "max_board": 40,                 # 允许的最大网格边长
    "skew_warn_deg": 0.7,            # 达到此倾斜角自动转正(度)
    "skew_max_deg": 8.0,             # 超过此倾斜角报错(度)
    "glyph_conf": 0.55,              # 数字字形置信度阈值(低于打印告警)
    "glyph_margin_ratio": 0.15,      # 首选/次选分差低于此值打印提示
    "glyph_min_area": 12,            # 数字笔画连通域最小面积(像素)
    "cell_ink_min_frac": 0.008,      # 判定格内有数字的最小墨水占比
    "template_size": 32,             # 数字模板归一化尺寸
    "use_template_cache": True,      # 模板缓存到固定文件 templates.npz

    # 求解
    "solve_time_budget": 60.0,       # 单题求解时间预算(秒)

    # 屏幕交互 / 作答
    "click_interval": 0.012,         # 相邻两次点击间隔(秒)
    "cell_delay": 0.015,             # 每 20 格点击后停顿(秒)
    "jitter": 1.5,                   # 点击位置随机抖动(像素)
    "submit_hotkey": "enter",        # 提交键
    "stop_hotkey": "esc",            # 作答中断热键
}


def recognize_and_solve(img):
    """对一张截图完成识别+求解. 返回 dict 结果(不含作答)."""
    res = {"ok": False, "stage": "", "msg": ""}
    try:
        puzzle = recognizer.recognize(img, CFG)
    except recognizer.RecognizeError as e:
        res["stage"] = "识别"
        res["msg"] = f"识别失败: {e}"
        return res
    except Exception as e:                       # 识别异常 → 明确报错可重试
        res["stage"] = "识别"
        res["msg"] = f"识别异常: {e}"
        print(f"[识别] {res['msg']}")
        return res

    for w in puzzle.warnings:
        print(f"[识别] {w}")

    t0 = time.time()
    status, grid, _info = sat_engine.solve_auto(
        puzzle.shape, puzzle.nums, time_budget=CFG["solve_time_budget"])
    elapsed = time.time() - t0

    if status != "ok":
        res["stage"] = "求解"
        if status == "unsat":
            res["msg"] = "题目无解(识别出的数字组合违反数墙规则, " \
                         "多半是数字识别有误, 请核对识别结果后重试)"
        else:
            res["msg"] = f"求解超时({elapsed:.0f}s), 可重试"
        return res

    print(f"[求解] 完成 ({elapsed:.2f}s)")
    res["ok"] = True
    res["puzzle"] = puzzle
    res["grid"] = grid
    return res


def run_real():
    automation.set_dpi_aware()
    stop = automation.StopFlag()
    stop.start(CFG["stop_hotkey"])
    try:
        while True:
            print("[流程] 请在屏幕上拖动框选整个数墙棋盘(框大一点没关系)...")
            bbox = automation.select_region()
            if bbox is None:
                ans = input("未完成框选。回车重新框选, 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue
            print(f"[流程] 框选区域: {bbox[0]} -> {bbox[1]}")
            img = automation.grab_screen(bbox)
            try:
                res = recognize_and_solve(img)
            except Exception as e:                # 任何意外都不静默退出
                print(f"[流程] 发生意外错误: {e}")
                ans = input("回车重试, 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue

            if not res["ok"]:
                print(f"[错误] [{res['stage']}] {res['msg']}")
                ans = input("回车重新框选重试, 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue

            puzzle = res["puzzle"]
            grid = res["grid"]
            print("识别到的题目与求解结果(#=黑 .=白):")
            print(solver.pretty(puzzle.shape, puzzle.nums, grid))

            print(f"[作答] 开始点击黑格(共 "
                  f"{sum(1 for r in range(len(puzzle.shape)) for c in range(len(puzzle.shape[0])) if grid[r][c])}"
                  f" 格), 按 {CFG['stop_hotkey'].upper()} 可中断...")
            try:
                clicked = automation.fill_answer(
                    puzzle, grid, bbox[0], CFG, stop)
            except automation.StopRequested:
                print("[作答] 已被用户中断, 未提交")
                ans = input("回车重新开始(重新框选), 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue
            print(f"[作答] 已点击 {clicked} 个黑格")
            automation.submit_answer(CFG)
            print("[流程] 完成")
            return
    finally:
        stop.cleanup()


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
    run_real()


if __name__ == "__main__":
    main()
