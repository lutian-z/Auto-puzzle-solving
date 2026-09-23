# -*- coding: utf-8 -*-
"""main.py — 马赛克(Mosaic)自动解题.

依赖: opencv-python numpy Pillow scipy mss pyautogui keyboard
      (scipy 缺失时自动退化为纯 Python 求解器, 大图可能超时)

用法:
    python main.py            # 框选→识别→求解→清残留→作答→提交

配置全部写死在本文件 CFG 中, 路径相对本文件所在目录解析, 无硬编码盘符。
流程中的异常(识别失败/求解超时/无解/作答失败/用户中断)都会给出明确
提示并允许重试, 不会静默崩溃; 全程 ESC 可中断作答。

作答语义: 点击 = 切换涂黑/未涂。解为黑的未涂格点一下涂黑; 棋盘上已有
黑格若解为白则点一下取消; 提交前回读棋面校验, 不一致不提交。
"""
import logging
import sys
import time

import automation
import recognizer
import sat_engine
import solver

# 配置(全部写死)
CFG = {
    "dark_v": 60,                    # 深色背景预裁的灰度阈值
    "min_cell_px": 12,               # 单元格最小边长, 过小判定非题目
    "max_board": 99,                 # 允许的最大网格边长(任意 N×N)
    "skew_warn_deg": 0.7,            # 达到此倾斜角自动转正(度)
    "skew_max_deg": 8.0,             # 超过此倾斜角报错(度)
    "spacing_cv_max": 0.12,          # 网格线间距变异系数上限
    "fake_grid_multi_frac": 0.3,     # 数字格多字形占比上限(假粗格盘侦测)
    "fake_grid_uniform": 0.62,       # 格内中值±12覆盖率下限(假粗格盘侦测)
    "glyph_conf": 0.55,              # 数字字形置信度阈值(低于触发变体重认)
    "glyph_min_area": 10,            # 数字笔画连通域最小面积(像素)
    "ink_frac_max": 0.5,             # 格内墨迹占比上限(超过判状态可疑)
    "calib_margin": 120,             # 锚点校准截图向四周扩大的边距(px)

    "solve_time_budget": 60.0,       # 单题求解时间预算(秒)

    "click_interval": 0.012,         # 相邻两次点击间隔(秒)
    "cell_delay": 0.015,             # 每 20 格点击后停顿(秒)
    "jitter": 1.5,                   # 点击位置随机抖动(像素)
    "submit_hotkey": "enter",        # 提交键
    "stop_hotkey": "esc",            # 作答中断热键
}


def setup_logging():
    """控制台日志(INFO)."""
    logger = logging.getLogger("mosaic")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)
    return logger


def recognize_and_solve(img, log):
    """对一张截图完成识别+求解+独立校验. 返回 dict 结果(不含作答)."""
    res = {"ok": False, "stage": "", "msg": ""}
    try:
        puzzle = recognizer.recognize(img, CFG, log)
    except recognizer.RecognitionError as e:
        res["stage"] = "识别"
        res["msg"] = str(e)
        return res
    except Exception as e:                       # 识别异常 → 明确报错可重试
        res["stage"] = "识别"
        res["msg"] = f"{type(e).__name__}: {e}"
        log.error("[识别] %s", res["msg"])
        return res

    for w in puzzle.warnings:
        log.warning("[识别告警] %s", w)

    t0 = time.time()
    try:
        status, grid, info = sat_engine.solve_auto(
            puzzle.n, puzzle.nums, time_budget=CFG["solve_time_budget"],
            log=log)
    except Exception as e:
        res["stage"] = "求解"
        res["msg"] = f"求解器异常: {type(e).__name__}: {e}"
        log.error("[求解] %s", res["msg"])
        return res
    elapsed = time.time() - t0

    if status != "ok":
        res["stage"] = "求解"
        if status == "unsat":
            res["msg"] = ("题目无解或多解(识别出的约束违反马赛克规则, "
                          "多半是数字/涂黑状态识别有误, 请重新框选)")
        else:
            res["msg"] = f"求解超时({elapsed:.0f}s, {puzzle.n}x{puzzle.n}), 可重试"
        return res

    vok, errs = solver.verify(puzzle.n, puzzle.nums, grid)
    if not vok:                                   # 双保险, 理论不可达
        res["stage"] = "校验"
        res["msg"] = "解未通过独立校验: " + "; ".join(errs[:3])
        log.error("[校验] %s", res["msg"])
        return res

    uniq = info.get("unique")
    if uniq is not True:
        res["stage"] = "求解"
        res["msg"] = ("识别出的题目未能证明唯一解, 与唯一解谜题不符, "
                      "识别可能有误, 请重新框选")
        return res
    log.info("[求解] 完成 (%.2fs, %s, 引擎=%s), 自检通过",
             elapsed, "已证明唯一解", info.get("engine", "?"))
    res["ok"] = True
    res["puzzle"] = puzzle
    res["grid"] = grid
    return res


def solution_black_cells(n, grid):
    """解中涂黑的格集合(含数字格 — 数字格自身也是涂黑候选)."""
    return {(r, c) for r in range(n) for c in range(n) if grid[r][c]}


def fmt_cells(cells):
    return " ".join(f"({r + 1},{c + 1})" for r, c in sorted(cells)[:10]) \
        or "无"


def run_real(log):
    automation.set_dpi_aware()
    stop = automation.StopFlag()
    stop.start(CFG["stop_hotkey"])
    try:
        while True:
            print("[流程] 请在屏幕上拖动框选整个马赛克棋盘...")
            bbox = automation.select_region()
            if bbox is None:
                ans = input("未完成框选。回车重新框选, 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue
            print(f"[流程] 框选区域: {bbox[0]} -> {bbox[1]}")
            img = automation.grab_screen(bbox)
            try:
                res = recognize_and_solve(img, log)
            except Exception as e:                # 任何意外都不静默退出
                log.error("[流程] 发生意外错误: %s", e)
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
            print(f"识别到的题目: {puzzle.describe()}")
            print(solver.pretty(puzzle.n, puzzle.nums, grid))

            want_black = solution_black_cells(puzzle.n, grid)
            have_black = set(puzzle.black_cells)
            to_white = have_black - want_black    # 已涂黑但解为白 → 点掉
            to_black = want_black - have_black    # 解为黑但未涂 → 点黑

            # 作答前锚点校准(窗口可能被移动)
            origin = automation.calibrate_anchor(bbox, puzzle, CFG, log)
            clear_plan = automation.plan_clicks(puzzle, to_white, origin)
            fill_plan = automation.plan_clicks(puzzle, to_black, origin)

            print(f"[作答] 按 {CFG['stop_hotkey'].upper()} 可中断...")
            try:
                if clear_plan:
                    print(f"[作答] 先清除棋盘上与解不符的 {len(clear_plan)} "
                          f"个已涂黑格...")
                    automation.fill_answer(clear_plan, CFG, stop, log)
                print(f"[作答] 点击 {len(fill_plan)} 个格子涂黑...")
                automation.fill_answer(fill_plan, CFG, stop, log)
            except automation.StopRequested:
                print("[作答] 已被用户中断, 未提交")
                ans = input("回车重新开始(重新框选), 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue
            except automation.InputSimulationError as e:
                print(f"[错误] [作答] {e}; 请检查窗口位置后重新框选")
                ans = input("回车重新框选, 输入 q 退出: ").strip().lower()
                if ans == "q":
                    return
                continue

            # 提交前回读校验: 一致或无法回读 → 直接提交;
            #      明确检测到棋面与解不一致 → 不提交, 提示后回框选 ----
            placed = None
            try:
                img2 = automation.grab_screen(bbox)
                placed = recognizer.read_black_cells(img2, CFG)
            except Exception:
                placed = None
            if placed is not None and placed[0] == puzzle.n:
                extra = sorted(placed[1] - want_black)
                missing = sorted(want_black - placed[1])
                if extra or missing:
                    print(f"[校验] 回读与解不一致(多黑: {fmt_cells(extra)}; "
                          f"缺黑: {fmt_cells(missing)}), 不提交, 请重新框选"
                          f"(必要时可先手动清空棋盘)")
                    continue

            automation.submit_answer(CFG, log)
            print("[流程] 完成")
            return
    finally:
        stop.cleanup()


def main():
    run_real(setup_logging())


if __name__ == "__main__":
    main()
