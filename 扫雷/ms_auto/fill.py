# -*- coding: utf-8 -*-
"""ms_auto.fill — 回填管线: 滚动逐行插旗(点选) 与 框内直接插旗(框选), 点击前逐行校验
"""
import time


from .desktop import (grab_screen, scroll_wheel, move_mouse, right_click,
                      _print)
from .grid import analyze_screen, _screens_identical
from .classify import classify_screen, FLAG
from .rows import (_use_reference_columns, _row_values_at, _fill_row_evidence,
                   _verify_row_against_board, _choose_fill_g0,
                   _fill_match_is_reliable, _choose_bottomup_fill_window)

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


# 框选模式回填: 框内直接插旗(全程不滚动)

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
        row_mine_cols = mines_by_row.get(row, [])
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
        # 每行点击前重新截屏校验(无滚动; 防页面渲染意外导致错位)。
        # 只截屏不再 analyze_screen: 页面静止, 格坐标来自识别网格;
        # 小格距下 analyze 单次 ~110ms, 每行白算一次是大头开销。
        img_r = grab_screen(bbox)
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

