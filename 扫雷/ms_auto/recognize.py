# -*- coding: utf-8 -*-
"""ms_auto.recognize — 识别管线: 滚动拼接(点选) 与 单屏识别(框选), 全程不猜底边
"""
import time

import numpy as np

from .desktop import (grab_screen, scroll_wheel, get_screen_size,
                      _print)
from .grid import (analyze_screen, compute_scroll_offset, _screens_identical,
                   refine_horizontal_bbox, _grab_analyze_region,
                   _anchor_board_top)
from .classify import classify_screen, UNOPENED
from .rows import (StitchedBoard, _normalize_row, _fuse_row_observations,
                   _choose_screen_g0, _screen_match_is_reliable,
                   _use_reference_columns)


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

    # 0/1. 定位棋盘(水平微调 + 尊重用户点选)
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

    # 2. 滚动拼接
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

    # 3. 组装全局棋盘
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


# 框选模式: 单屏识别(题目完整可见, 全程不滚动)

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

    def _build_board(vals):
        bd = np.zeros((n, n), dtype=int)
        for r in range(n):
            row = vals[r]
            if row.shape[0] > n:
                row = row[:n]
            elif row.shape[0] < n:
                row = np.concatenate(
                    [row, np.full(n - row.shape[0], -1, dtype=int)])
            bd[r] = row
        return bd

    def _count_contradictions(bd):
        import ms_solver as _ms
        cnt = 0
        for r in range(n):
            for c in range(n):
                v = int(bd[r, c])
                if v <= 0:
                    continue
                unop = sum(1 for dr, dc in _ms.NEIGHBORS
                           if 0 <= r + dr < n and 0 <= c + dc < n
                           and bd[r + dr, c + dc] in (-1, -2))
                if unop < v:
                    cnt += 1
        return cnt

    # 自适应救援: 若自洽检查有矛盾(个别未翻开格的浮雕高光被渲染相位
    # 吞掉), 放宽一档结构判据重分类一次; 只有矛盾严格减少才采用, 否则
    # 维持原结果由调用方安全终止。判据尺度无关, 任何格距都适用。
    from . import classify as _cls
    if _count_contradictions(_build_board(values)) > 0:
        saved = dict(_cls.UNOPENED_PARAMS)
        _cls.UNOPENED_PARAMS.update(
            {"thr_delta": 6.0, "band_div": 4, "band_cap": 10, "ratio": 0.45})
        try:
            values2, details2, _ = classify_screen(
                img, grid, templates=config.get("templates"))
        finally:
            _cls.UNOPENED_PARAMS.clear()
            _cls.UNOPENED_PARAMS.update(saved)
        if _count_contradictions(_build_board(values2)) < \
                _count_contradictions(_build_board(values)):
            _print("[识别] 高光救援: 采用放宽结构判据的二次分类")
            values, details = values2, details2

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

