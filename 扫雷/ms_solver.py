# -*- coding: utf-8 -*-
"""ms_solver.py — 扫雷求解器.

输入棋盘值含义:
  -2  已标记的雷(旗, 来自回填/人工)
  -1  未翻开(可能是雷, 也可能是安全的未翻开格)
   0-8 翻开数字(周围雷数)

输出判定数组:
  MINE(-2)  确定是雷
  SAFE(1)   确定安全(非雷)
  UNKNOWN(0) 无法判定
  OPENED(-3) 已翻开格

面向"静态题目": 看到的数字就是全部信息, 点开不会出现新数字,
必须一次性从初始数字推出所有确定雷.

算法(分层自适应):
  1. 全棋盘精确整数约束 —— 每个未翻开格为0/1变量, 每个数字为等式;
     求出一组答案后排除它再求一次, 从而证明唯一性而非仅找到可行解.
  2. 真多解时逐格取反求可行性, 只输出所有解都一致的格子.
  3. SciPy不可用或精确后端超时时, 后备使用基础传播、子集差集推理、
     组件枚举和传播反证, 仍不猜测.

全程受 time_limit 约束, 超时即返回已确定的结论(保证不会卡死).
无法确定的格保持 UNKNOWN, 回填阶段只标记确定雷(不误判).

全棋盘建模不依赖边长、5的倍数、组件形状或局部解数量.
"""
import sys
import numpy as np
from collections import defaultdict

sys.setrecursionlimit(200000)

MINE = -2
SAFE = 1
UNKNOWN = 0
OPENED = -3
UNOPENED = -1     # 输入中的未翻开格
FLAG = -2          # 输入中的已标记雷

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]


class InconsistentBoardError(ValueError):
    """识别棋盘或求解结论违反至少一个数字约束."""

    def __init__(self, violations=None, message=None):
        self.violations = list(violations or [])
        preview = ", ".join(
            f"({r + 1},{c + 1})={num}, 已判雷{mines}, 未知{unknown}"
            for r, c, num, mines, unknown in self.violations[:5]
        )
        detail = message or preview or "全局约束无可行解"
        super().__init__(f"扫雷约束矛盾: {detail}")


def _neighbors(r, c, n):
    for dr, dc in NEIGHBORS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


def _neighbor_count(mask):
    """n×n 布尔 → n×n 整数: out[r,c] = mask 中 (r,c) 的 8 邻域 True 个数."""
    n = mask.shape[0]
    out = np.zeros((n, n), dtype=int)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            # 目标 (r,c) 的邻居在 (r+dr, c+dc); 只需合法范围内累加
            r0, r1 = max(0, -dr), min(n, n - dr)
            c0, c1 = max(0, -dc), min(n, n - dc)
            if r1 <= r0 or c1 <= c0:
                continue
            dest_r = np.arange(r0, r1)
            src_r = dest_r + dr
            dest_c = np.arange(c0, c1)
            src_c = dest_c + dc
            out[np.ix_(dest_r, dest_c)] += mask[np.ix_(src_r, src_c)]
    return out


def constraint_violations(board, result):
    """返回求解结论中不可能满足的数字约束.

    每项为 ``(行, 列, 数字, 已判雷数, 剩余未知数)``。即使求解因预算
    留下 UNKNOWN, 也应始终满足 ``已判雷 <= 数字 <= 已判雷 + UNKNOWN``；
    否则说明识别数据有误或推理过程产生了互相冲突的结论.
    """
    if board.shape != result.shape:
        raise ValueError("board 与 result 尺寸不一致")
    n = board.shape[0]
    digits = board >= 0
    is_mine = (board == FLAG) | (result == MINE)
    is_unknown = (board == UNOPENED) & (result == UNKNOWN)
    mines_nb = _neighbor_count(is_mine)
    unknown_nb = _neighbor_count(is_unknown)
    nums = np.where(digits, board, 0).astype(int)
    bad = digits & ((mines_nb > nums) | (mines_nb + unknown_nb < nums))
    rs, cs = np.nonzero(bad)
    return [(int(r), int(c), int(board[r, c]),
             int(mines_nb[r, c]), int(unknown_nb[r, c]))
            for r, c in zip(rs, cs)]


# 基础传播

def basic_propagate(board, unknowns, status):
    """基础传播. status: 1=雷 -1=安全 0=未知. 返回是否变化."""
    n = board.shape[0]
    changed = False
    for r in range(n):
        for c in range(n):
            num = board[r, c]
            if num < 0:
                continue
            nb = list(_neighbors(r, c, n))
            unop = [p for p in nb if unknowns[p] and status[p] == 0]
            flagged = sum(1 for p in nb if unknowns[p] and
                          (status[p] == 1 or board[p] == FLAG))
            if num == 0:
                for p in unop:
                    if status[p] != -1:
                        status[p] = -1
                        changed = True
                continue
            if not unop:
                continue
            remaining = num - flagged
            if remaining < 0:
                continue
            if remaining == 0:
                for p in unop:
                    if status[p] != -1:
                        status[p] = -1
                        changed = True
            elif len(unop) == remaining:
                for p in unop:
                    if status[p] != 1:
                        status[p] = 1
                        changed = True
    return changed


# 约束收集

def _collect_constraints(board, unknowns, status):
    """收集边界约束: [(need, frozenset(未判未知格))]."""
    n = board.shape[0]
    constraints = []
    seen = set()
    for r in range(n):
        for c in range(n):
            num = board[r, c]
            if num <= 0:
                continue
            nb = list(_neighbors(r, c, n))
            unop = [p for p in nb if unknowns[p] and status[p] == 0]
            if not unop:
                continue
            flagged = sum(1 for p in nb if unknowns[p] and
                          (status[p] == 1 or board[p] == FLAG))
            need = num - flagged
            if need <= 0 or need >= len(unop):
                continue
            key = (need, frozenset(unop))
            if key in seen:
                continue
            seen.add(key)
            constraints.append((need, frozenset(unop)))
    return constraints


def _propagate_constraints(constraints, status):
    """在约束集合上传播(就地修改 status). 返回 False 若矛盾."""
    changed = True
    while changed:
        changed = False
        for need, cells in constraints:
            un = [p for p in cells if status.get(p, 0) == 0]
            if not un:
                if sum(1 for p in cells if status.get(p, 0) == 1) != need:
                    return False
                continue
            flagged = sum(1 for p in cells if status.get(p, 0) == 1)
            rem = need - flagged
            if rem < 0:
                return False
            if rem == 0:
                for p in un:
                    if status.get(p, 0) != -1:
                        status[p] = -1
                        changed = True
            elif rem == len(un):
                for p in un:
                    if status.get(p, 0) != 1:
                        status[p] = 1
                        changed = True
    return True


def subset_deduction(board, unknowns, status):
    """子集差集推理. 返回是否变化."""
    constraints = _collect_constraints(board, unknowns, status)
    changed = False
    L = len(constraints)
    for i in range(L):
        need_i, set_i = constraints[i]
        for j in range(i + 1, L):
            need_j, set_j = constraints[j]
            if set_i <= set_j:
                diff = set_j - set_i
                if not diff:
                    continue
                dneed = need_j - need_i
                if dneed == 0:
                    for p in diff:
                        if status[p] != -1:
                            status[p] = -1
                            changed = True
                elif dneed == len(diff):
                    for p in diff:
                        if status[p] != 1:
                            status[p] = 1
                            changed = True
            elif set_j <= set_i:
                diff = set_i - set_j
                if not diff:
                    continue
                dneed = need_i - need_j
                if dneed == 0:
                    for p in diff:
                        if status[p] != -1:
                            status[p] = -1
                            changed = True
                elif dneed == len(diff):
                    for p in diff:
                        if status[p] != 1:
                            status[p] = 1
                            changed = True
    return changed


def run_deduction(board, unknowns, status):
    """阶段1+2交替传播直到稳定."""
    changed = True
    while changed:
        changed = False
        while basic_propagate(board, unknowns, status):
            changed = True
        if subset_deduction(board, unknowns, status):
            changed = True


# 组件分解

def _decompose(constraints):
    """按共享格把约束分成连通组件."""
    adj = defaultdict(set)
    cons = list(constraints)
    L = len(cons)
    for i in range(L):
        for j in range(i + 1, L):
            if cons[i][1] & cons[j][1]:
                adj[i].add(j)
                adj[j].add(i)
    seen = set()
    comps = []
    for i in range(L):
        if i in seen:
            continue
        comp = []
        q = [i]
        seen.add(i)
        while q:
            cur = q.pop()
            comp.append(cons[cur])
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


# 完整枚举

def _enumerate_component(comp, max_sols=20000, node_budget=80000000,
                         time_limit=5.0):
    """枚举一个连通组件的全部可行解.

    返回 (solutions, exhausted).
    """
    import time as _time
    _t0 = _time.time()
    var_set = set()
    for _, cells in comp:
        var_set.update(cells)
    var_list = sorted(var_set)

    sols = []
    status = {p: 0 for p in var_list}
    cnt = {"n": 0, "hit": False}

    def prop():
        changed = True
        while changed:
            changed = False
            for num, cells in comp:
                un = [p for p in cells if status[p] == 0]
                if not un:
                    if sum(1 for p in cells if status[p] == 1) != num:
                        return False
                    continue
                fl = sum(1 for p in cells if status[p] == 1)
                rem = num - fl
                if rem < 0:
                    return False
                if rem == 0:
                    for p in un:
                        status[p] = -1
                        changed = True
                elif rem == len(un):
                    for p in un:
                        status[p] = 1
                        changed = True
        return True

    def consistent():
        for num, cells in comp:
            un = [p for p in cells if status[p] == 0]
            fl = sum(1 for p in cells if status[p] == 1)
            rem = num - fl
            if rem < 0 or rem > len(un):
                return False
        return True

    def dfs():
        cnt["n"] += 1
        if cnt["n"] > node_budget:
            cnt["hit"] = True
            return
        if _time.time() - _t0 > time_limit:
            cnt["hit"] = True
            return
        if len(sols) >= max_sols:
            cnt["hit"] = True
            return
        snap = dict(status)
        if not prop():
            status.clear()
            status.update(snap)
            return
        cand = [p for p in var_list if status[p] == 0]
        if not cand:
            sols.append(set(p for p in var_list if status[p] == 1))
            status.clear()
            status.update(snap)
            return
        best = None
        for p in cand:
            tight = 0
            for num, cells in comp:
                if p in cells and status[p] == 0:
                    tight += num - sum(1 for q in cells if status[q] == 1)
            if best is None or tight < best[1]:
                best = (p, tight)
        p = best[0]
        for v in (1, -1):
            status[p] = v
            if consistent():
                dfs()
            status[p] = 0
            if len(sols) >= max_sols:
                break
        status.clear()
        status.update(snap)

    dfs()
    exhausted = (not cnt["hit"]) and len(sols) < max_sols
    return sols, exhausted


# 传播反证兜底

def _propagate_contradiction(comp, var_list, base, pending):
    """传播反证: 对每个待定格, 假设雷/安全, 传播看是否矛盾."""
    mines, safes = set(), set()
    for p in pending:
        s1 = dict(base)
        s1[p] = 1
        ok1 = _propagate_constraints(comp, s1)
        s2 = dict(base)
        s2[p] = -1
        ok2 = _propagate_constraints(comp, s2)
        if not ok1 and ok2:
            safes.add(p)
        if not ok2 and ok1:
            mines.add(p)
    return mines, safes


# 全棋盘精确整数约束

def _solve_global_milp(board, time_limit=120.0, verbose=False):
    """用全棋盘 0/1 整数约束求解并证明唯一性.

    每个未翻开格是一个二元变量，每个数字直接变成相邻变量之和的等式。
    先求一个可行解，再加入一条排除该完整解的约束：若第二次无解，则已从
    数学上证明唯一，整盘可直接判定。此方法不枚举海量局部组合，也不依赖
    棋盘边长、数字分布或固定组件大小。

    SciPy 不可用或求解器在预算内未返回结论时返回 None，由旧逻辑安全后备。
    若题目确有多解，则逐格反证，只返回在所有解中都相同的格子。
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except ImportError:
        return None

    import time as _time
    t0 = _time.monotonic()
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("扫雷棋盘必须是方阵")
    n = board.shape[0]

    # 用 numpy 一次性构造约束(替代逐格 Python 循环)
    u_rows, u_cols = np.nonzero(board == UNOPENED)
    m = len(u_rows)
    cells = list(zip(u_rows.tolist(), u_cols.tolist()))
    col_of = np.full((n, n), -1, dtype=np.int64)
    col_of[u_rows, u_cols] = np.arange(m)

    offs = np.array(NEIGHBORS, dtype=np.int64)
    dr, dc = np.nonzero(board >= 0)
    nd = len(dr)
    nbr_r = dr[:, None] + offs[None, :, 0]
    nbr_c = dc[:, None] + offs[None, :, 1]
    valid = (nbr_r >= 0) & (nbr_r < n) & (nbr_c >= 0) & (nbr_c < n)
    rr = np.clip(nbr_r, 0, n - 1)
    cc = np.clip(nbr_c, 0, n - 1)
    nb_col = np.where(valid, col_of[rr, cc], -1)          # 未翻开邻格的列号
    is_flag = np.where(valid, board[rr, cc] == FLAG, False)
    flagged_cnt = is_flag.sum(axis=1)
    needs = board[dr, dc].astype(np.int64) - flagged_cnt
    uni = (nb_col >= 0) & (~is_flag)                      # 参与约束的未翻开邻格
    counts = uni.sum(axis=1)

    row_ids = []
    col_ids = []
    data = []
    rhs = []
    local_bad = []
    sel_digits = []
    row_no = 0
    for k in range(nd):
        if needs[k] < 0 or needs[k] > counts[k]:
            local_bad.append((int(dr[k]), int(dc[k]), int(board[dr[k], dc[k]]),
                              int(flagged_cnt[k]), int(counts[k])))
            continue
        if counts[k] == 0:
            if needs[k] != 0:
                local_bad.append((int(dr[k]), int(dc[k]), int(board[dr[k], dc[k]]),
                                  int(flagged_cnt[k]), 0))
            continue
        sel_digits.append(k)
        row_ids.extend([row_no] * int(counts[k]))
        rhs.append(float(needs[k]))
        row_no += 1
    if local_bad:
        raise InconsistentBoardError(local_bad)
    if sel_digits:
        sel = np.asarray(sel_digits, dtype=np.int64)
        col_ids = nb_col[sel][uni[sel]].tolist()
        data = [1.0] * len(col_ids)

    result = np.full(board.shape, OPENED, dtype=int)
    result[board == FLAG] = MINE
    if m == 0:
        return result

    constraints = []
    if row_no:
        matrix = csr_matrix((data, (row_ids, col_ids)),
                            shape=(row_no, m), dtype=float)
        target = np.asarray(rhs, dtype=float)
        constraints.append(LinearConstraint(matrix, target, target))

    objective = np.zeros(m, dtype=float)
    integrality = np.ones(m, dtype=int)
    bounds = Bounds(np.zeros(m), np.ones(m))

    def remaining_time():
        return max(0.05, float(time_limit) - (_time.monotonic() - t0))

    first = milp(objective, integrality=integrality, bounds=bounds,
                 constraints=constraints,
                 options={"time_limit": remaining_time(), "mip_rel_gap": 0.0})
    if not first.success:
        if first.status == 2:  # HiGHS: infeasible
            raise InconsistentBoardError(
                message="全部数字合并后无可行解，题面至少有一格识别错误")
        return None
    reference = np.rint(first.x).astype(int)

    # 排除 reference：sum(不同于reference的位) >= 1。
    coeff = np.where(reference == 1, -1.0, 1.0)
    lower = 1.0 - float(reference.sum())
    no_good = LinearConstraint(
        csr_matrix(coeff.reshape(1, -1)),
        np.asarray([lower]), np.asarray([np.inf]))
    second = milp(objective, integrality=integrality, bounds=bounds,
                  constraints=constraints + [no_good],
                  options={"time_limit": remaining_time(), "mip_rel_gap": 0.0})

    if not second.success and second.status == 2:
        for i, (r, c) in enumerate(cells):
            result[r, c] = MINE if reference[i] else SAFE
        if verbose:
            print("[求解] 全棋盘整数约束已证明唯一解")
        return result
    if not second.success:
        return None

    # 多解安全路径：第二解中已经变化的格必不确定；其余格逐一尝试取反。
    alternate = np.rint(second.x).astype(int)
    ambiguous = set(np.flatnonzero(alternate != reference).tolist())
    forced = set()
    pending = [i for i in range(m) if i not in ambiguous]
    for i in pending:
        if remaining_time() <= 0.06:
            break
        unit = csr_matrix(([1.0], ([0], [i])), shape=(1, m))
        opposite = 1.0 - float(reference[i])
        fixed = LinearConstraint(unit, np.asarray([opposite]),
                                 np.asarray([opposite]))
        probe = milp(objective, integrality=integrality, bounds=bounds,
                     constraints=constraints + [fixed],
                     options={"time_limit": remaining_time(),
                              "mip_rel_gap": 0.0})
        if not probe.success and probe.status == 2:
            forced.add(i)
        elif probe.success:
            other = np.rint(probe.x).astype(int)
            ambiguous.update(np.flatnonzero(other != reference).tolist())
    for i, (r, c) in enumerate(cells):
        if i in forced:
            result[r, c] = MINE if reference[i] else SAFE
        else:
            result[r, c] = UNKNOWN
    if verbose:
        print("[求解] 题面存在多个可行解，仅保留逐格反证确定的结论")
    return result


# 主入口

def solve_board(board, verbose=False,
                max_solutions=50000, node_budget=300000000,
                time_limit=120.0):
    """求解扫雷棋盘.

    board: n×n int array, -1 未翻开, -2 旗, 0-8 数字.
    返回 n×n 判定数组: MINE(-2)/SAFE(1)/UNKNOWN(0)/OPENED(-3).

    优先把全棋盘一次性建成 0/1 整数约束，并通过排除第一组完整答案证明
    唯一性。若运行环境没有 SciPy 或精确求解超时，再使用原有的传播、子集、
    组件枚举和反证作为安全后备。
    """
    import time as _time

    exact = _solve_global_milp(board, time_limit=time_limit, verbose=verbose)
    if exact is not None:
        violations = constraint_violations(board, exact)
        if violations:
            raise InconsistentBoardError(violations)
        if verbose:
            nm = int((exact == MINE).sum())
            nu = int((exact == UNKNOWN).sum())
            print(f"[求解] 雷 {nm}, 安全 {int((exact == SAFE).sum())}, 未知 {nu}")
        return exact

    t0 = _time.time()

    n = board.shape[0]
    opened = board >= 0
    unknowns = board == -1
    status = np.zeros((n, n), dtype=int)

    sol_mine = set()
    sol_safe = set()

    for _round in range(20):
        if _time.time() - t0 > time_limit:
            break

        # 阶段1: 确定性传播 + 子集推理
        run_deduction(board, unknowns, status)

        constraints = _collect_constraints(board, unknowns, status)
        if not constraints:
            break

        new_mine = set()
        new_safe = set()

        for comp in _decompose(constraints):
            if _time.time() - t0 > time_limit:
                break
            var_list = sorted(set().union(*[cells for _, cells in comp]))
            base = {p: status[p] for p in var_list if status[p] != 0}
            pending = [p for p in var_list if status[p] == 0]
            if not pending:
                continue

            # 完整枚举: 唯一解/少解时能快速穷尽, 得到确定结论.
            # 用户题目都是唯一解, 完整枚举能求全. 组件级时间预算防止卡死.
            comp_tlim = max(3.0, min(15.0, time_limit / max(1, len(_decompose(constraints)))))
            sols, exhausted = _enumerate_component(comp, max_sols=max_solutions,
                                                   node_budget=node_budget,
                                                   time_limit=comp_tlim)
            if exhausted and sols:
                ns = len(sols)
                var_set = set()
                for _, cells in comp:
                    var_set.update(cells)
                mine_cnt = {p: 0 for p in var_set}
                for s in sols:
                    for p in s:
                        mine_cnt[p] += 1
                for p, v in mine_cnt.items():
                    if v == ns:
                        new_mine.add(p)
                    elif v == 0:
                        new_safe.add(p)
                continue

            # 完整枚举超时 -> 传播反证兜底
            m, s = _propagate_contradiction(comp, var_list, base, pending)
            new_mine |= m
            new_safe |= s

        # 阶段2: 反馈新结论到 status
        progressed = False
        for p in new_mine:
            if status[p] != 1:
                status[p] = 1
                sol_mine.add(p)
                progressed = True
        for p in new_safe:
            if status[p] != -1:
                status[p] = -1
                sol_safe.add(p)
                progressed = True

        if not progressed:
            break

    result = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            if opened[r, c]:
                result[r, c] = OPENED
            elif status[r, c] == 1 or (r, c) in sol_mine or board[r, c] == FLAG:
                result[r, c] = MINE
            elif status[r, c] == -1 or (r, c) in sol_safe:
                result[r, c] = SAFE
            else:
                result[r, c] = UNKNOWN

    violations = constraint_violations(board, result)
    if violations:
        raise InconsistentBoardError(violations)

    if verbose:
        nm = int((result == MINE).sum())
        nu = int((result == UNKNOWN).sum())
        print(f"[求解] 雷 {nm}, 安全 {int((result == SAFE).sum())}, 未知 {nu}")
    return result
