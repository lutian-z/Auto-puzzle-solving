# -*- coding: utf-8 -*-
"""ms_auto.rows — 行级数据结构与跨屏行匹配: 拼接棋盘 / 行证据 / 绝对行号选择
"""
import numpy as np

from .classify import UNOPENED, FLAG, classify_cell

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
    y1 = min(img_bgr.shape[0], y0 + int(round(pitch)))
    for c in range(n):
        x0, x1 = int(ref_vx[c]), int(ref_vx[c + 1])
        # 整格切片, 与 classify_screen 同一判据口径(尺度无关)
        cell = img_bgr[y0:y1, x0:x1]
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

