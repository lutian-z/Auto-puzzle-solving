# -*- coding: utf-8 -*-
"""main.py — 帐篷与树(Tents and Trees)自动解题.

依赖: opencv-python numpy Pillow mss pyautogui keyboard ortools
      (ortools 缺失时自动退化为纯 Python 求解器, 25x25 以上大图可能超时)

用法:
    python main.py            # 框选→识别→求解→清残留→作答→提交

配置全部写死在本文件 CFG 中, 路径相对本文件所在目录解析, 无硬编码盘符。
流程中的异常(识别失败/求解超时/无解/作答失败/用户中断)都会给出明确
提示并允许重试, 不会静默崩溃; 全程 ESC 可中断作答。
"""
import logging
import sys
import time

import automation
import recognizer
import sat_engine
import solver

# ----------------------------------------------------------------------
# 配置(全部写死)
# ----------------------------------------------------------------------
CFG = {
    "dark_v": 128,                   # 深色背景预裁的 V 阈值
    "min_cell_px": 12,               # 单元格最小边长, 过小判定非题目
    "max_board": 40,                 # 允许的最大网格边长
    "skew_warn_deg": 0.7,            # 达到此倾斜角自动转正(度)
    "skew_max_deg": 8.0,             # 超过此倾斜角报错(度)
    "spacing_cv_max": 0.12,          # 网格线间距变异系数上限
    "canopy_hi": 0.12,               # 树冠占比 ≥ 此值判树
    "canopy_lo": 0.05,               # 树冠占比灰区下限(灰区判树并告警)
    "glyph_conf": 0.55,              # 数字字形置信度阈值(低于触发变体重认)
    "glyph_min_area": 10,            # 数字笔画连通域最小面积(像素)
    "tent_dark_frac": 0.04,          # 格内暗像素占比超过此值判为帐篷(残留检测/回读)
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
    logger = logging.getLogger("tents")
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
            puzzle.n, sorted(puzzle.trees), puzzle.row_ct, puzzle.col_ct,
            time_budget=CFG["solve_time_budget"], log=log)
    except Exception as e:
        res["stage"] = "求解"
        res["msg"] = f"求解器异常: {type(e).__name__}: {e}"
        log.error("[求解] %s", res["msg"])
        return res
    elapsed = time.time() - t0

    if status != "ok":
        res["stage"] = "求解"
        if status == "unsat":
            res["msg"] = ("题目无解(识别出的约束违反帐篷规则, "
                          "多半是数字/树识别有误, 请重新框选)")
        else:
            res["msg"] = f"求解超时({elapsed:.0f}s, {puzzle.n}x{puzzle.n}), 可重试"
        return res

    vok, errs = solver.verify(puzzle.n, sorted(puzzle.trees),
                              puzzle.row_ct, puzzle.col_ct, grid)
    if not vok:                                   # 双保险, 理论不可达
        res["stage"] = "校验"
        res["msg"] = "解未通过独立校验: " + "; ".join(errs[:3])
        log.error("[校验] %s", res["msg"])
        return res

    uniq = info.get("unique")
    if uniq is False:
        res["stage"] = "求解"
        res["msg"] = ("识别出的题目存在多个合法解, 与唯一解谜题不符, "
                      "识别可能有误, 请重新框选")
        return res
    log.info("[求解] 完成 (%.2fs, %s, 引擎=%s), 自检通过",
             elapsed,
             "唯一解" if uniq else "未确认唯一性",
             info.get("engine", "?"))
    res["ok"] = True
    res["puzzle"] = puzzle
    res["grid"] = grid
    return res


def run_real(log):
    automation.set_dpi_aware()
    stop = automation.StopFlag()
    stop.start(CFG["stop_hotkey"])
    try:
        while True:
            print("[流程] 请在屏幕上拖动框选整个帐篷棋盘(含上方与左侧数字)...")
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
            print(solver.pretty(puzzle.n, sorted(puzzle.trees),
                                puzzle.row_ct, puzzle.col_ct, grid))

            # 作答前锚点校准(窗口可能被移动)
            origin = automation.calibrate_anchor(bbox, puzzle, CFG, log)
            solution_cells = sorted((r, c) for r in range(puzzle.n)
                                    for c in range(puzzle.n) if grid[r][c])
            clear_cells = sorted(puzzle.existing_tents)
            plan = automation.plan_clicks(puzzle, solution_cells, origin)
            clear_plan = automation.plan_clicks(puzzle, clear_cells, origin)

            print(f"[作答] 按 {CFG['stop_hotkey'].upper()} 可中断...")
            try:
                if clear_plan:
                    print(f"[作答] 清除棋盘上残留的 {len(clear_plan)} 个帐篷"
                          f"(非本局内容)...")
                    automation.fill_answer(clear_plan, CFG, stop, log)
                print(f"[作答] 点击 {len(plan)} 个帐篷格...")
                automation.fill_answer(plan, CFG, stop, log)
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

            # ---- 提交前回读校验: 一致或无法回读 → 直接提交;
            #      明确检测到棋面与解不一致 → 不提交, 提示后回框选 ----
            time.sleep(0.4)
            placed = None
            try:
                img2 = automation.grab_screen(bbox)
                placed = recognizer.read_tent_cells(img2, CFG)
            except Exception:
                placed = None
            if placed is not None and placed[0] == puzzle.n:
                extra = sorted(placed[1] - set(solution_cells))
                missing = sorted(set(solution_cells) - placed[1])
                if extra or missing:
                    def fmt(cells):
                        return " ".join(f"({r+1},{c+1})"
                                        for r, c in cells[:10]) or "无"
                    print(f"[校验] 回读与解不一致(多出: {fmt(extra)}; "
                          f"缺失: {fmt(missing)}), 不提交, 请重新框选"
                          f"(可先手动清空棋盘)")
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
