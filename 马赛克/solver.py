# -*- coding: utf-8 -*-
"""solver.py — 马赛克(Mosaic) 纯 Python 求解器.

规则建模:
    n×n 网格, 每格为 0/1 变量(1=涂黑); 数字格 (r,c)=k 表示以其为中心的
    3×3 邻域(含自身, 越界不计, 角4/边6/中9格)内涂黑格数恰为 k.
    与扫雷的关键差异: 数字格自身也是可涂黑变量(扫雷数字格必非雷),
    因此全部 n² 格都是未知变量, 无任何先验固定格.

对外接口(与帐篷项目 solver.py 同风格):
    solve(n, nums, ...)              -> (grid, info) 或抛异常
    verify(n, nums, grid)            -> (ok, errs) 独立校验
    pretty(n, nums, grid)            -> 控制台可读文本

算法: 约束传播(邻域计数上下界) + 子集差集推理 + 回溯(最少未知数字格优先),
搜索时最多收集 2 个解用于多解检测(题目保证唯一解, 找到即意义明确).
作为 sat_engine(scipy MILP 精确引擎)不可用/超时时的兜底, 不猜测.
"""
import sys
import time

# ----------------------------------------------------------------------
# 异常体系(所有失败给出可读信息并可重试)
# ----------------------------------------------------------------------

class MosaicError(Exception):
    """马赛克项目所有异常的基类."""


class NoSolutionError(MosaicError):
    """题目无解(通常意味着识别有误)."""


class MultiSolutionError(MosaicError):
    """题目存在多个合法解(唯一解谜题出现多解 → 识别大概率有误)."""


class SolverTimeoutError(MosaicError):
    """求解超出时间预算."""


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

NEIGH9 = [(-1, -1), (-1, 0), (-1, 1),
          (0, -1), (0, 0), (0, 1),
          (1, -1), (1, 0), (1, 1)]


def neighborhood(n, r, c):
    """(r,c) 的 3×3 邻域格列表(含自身, 裁掉越界)."""
    return [(r + dr, c + dc) for dr, dc in NEIGH9
            if 0 <= r + dr < n and 0 <= c + dc < n]


def build_constraints(n, nums):
    """把数字矩阵转成约束列表 [(中心, need, [邻域格])]."""
    cons = []
    for (r, c), k in sorted(nums.items()):
        if not (0 <= r < n and 0 <= c < n):
            raise MosaicError(f"数字格坐标越界: ({r},{c})")
        if not (0 <= k <= len(neighborhood(n, r, c))):
            raise MosaicError(
                f"数字 ({r+1},{c+1})={k} 超出邻域大小, 题面非法")
        cons.append(((r, c), k, neighborhood(n, r, c)))
    return cons


def _validate_input(n, nums):
    if not isinstance(n, int) or n < 1:
        raise MosaicError(f"非法网格规模 N={n}")
    for (r, c) in nums:
        if not (0 <= r < n and 0 <= c < n):
            raise MosaicError(f"数字格坐标越界: ({r},{c})")


# ----------------------------------------------------------------------
# 约束传播
# ----------------------------------------------------------------------

def _cell_constraints(cons, n):
    """格 -> 引用它的约束下标列表(值变化时增量触发)."""
    occ = {}
    for ci, (_pos, _k, cells) in enumerate(cons):
        for p in cells:
            occ.setdefault(p, []).append(ci)
    return occ


def _propagate_cons(cons, val, queue, occ):
    """计数传播至不动点. val: dict 格 -> -1/0/1. 返回 False=矛盾.

    对每条约束: 已黑 black, 未定 unk.
      black > need 或 black+|unk| < need → 矛盾;
      black == need → unk 全白; black+|unk| == need → unk 全黑.
    """
    in_queue = set(queue)
    while queue:
        p = queue.pop()
        in_queue.discard(p)
        for ci in occ.get(p, ()):
            _pos, need, cells = cons[ci]
            black = 0
            unk = []
            for q in cells:
                v = val[q]
                if v == 1:
                    black += 1
                elif v == -1:
                    unk.append(q)
            if black > need or black + len(unk) < need:
                return False
            if not unk:
                continue
            if black == need:
                newv = 0
            elif black + len(unk) == need:
                newv = 1
            else:
                continue
            for q in unk:
                val[q] = newv
                if q not in in_queue:
                    in_queue.add(q)
                    queue.append(q)
    return True


def _collect_bounds(cons, val):
    """收集活跃约束 [(need-black, frozenset(未定格))] 供子集推理."""
    out = []
    for _pos, need, cells in cons:
        black = 0
        unk = []
        for q in cells:
            v = val.get(q, -1)
            if v == 1:
                black += 1
            elif v == -1:
                unk.append(q)
        if not unk:
            continue
        rem = need - black
        if rem <= 0 or rem >= len(unk):
            # rem==0 / rem==|unk| 已被计数传播处理, 此处跳过纯冗余
            continue
        out.append((rem, frozenset(unk)))
    return out


def _subset_deduction(cons, val, queue):
    """子集差集推理: A⊆B → need_B-need_A 差约束作用在 B-A 上."""
    constraints = _collect_bounds(cons, val)
    changed = False
    L = len(constraints)
    for i in range(L):
        need_i, set_i = constraints[i]
        for j in range(i + 1, L):
            need_j, set_j = constraints[j]
            if set_i <= set_j:
                diff, dneed = set_j - set_i, need_j - need_i
            elif set_j <= set_i:
                diff, dneed = set_i - set_j, need_i - need_j
            else:
                continue
            if not diff:
                continue
            if dneed == 0:
                for p in diff:
                    if val.get(p, -1) == -1:
                        val[p] = 0
                        queue.append(p)
                        changed = True
            elif dneed == len(diff):
                for p in diff:
                    if val.get(p, -1) == -1:
                        val[p] = 1
                        queue.append(p)
                        changed = True
    return changed


# ----------------------------------------------------------------------
# 回溯搜索
# ----------------------------------------------------------------------

def _branch_cell(cons, val):
    """分支格选择: 未定格最少的活跃约束里取一个未定格(最受限优先)."""
    best = None
    best_unk = None
    for _pos, _need, cells in cons:
        unk = [q for q in cells if val.get(q, -1) == -1]
        if unk and (best_unk is None or len(unk) < len(best_unk)):
            best_unk = unk
            best = unk[0]
            if len(unk) == 2:
                break
    if best is None:                       # 防御: 理论不可达
        for p in sorted(val):
            if val[p] == -1:
                return p
        return None
    return best


def _search(cons, val, occ, deadline, solutions, seen, queue, limit=2):
    """回溯搜索, 收集最多 limit 个完整解(用于唯一性判定).

    queue: 值刚发生变化的格(增量传播种子); 首次调用传全部数字格中心,
    分支赋值后传该格 — 事件驱动的传播只在"有格变化"时才有意义.
    """
    if time.time() > deadline:
        raise SolverTimeoutError("搜索超时")
    if not _propagate_cons(cons, val, queue, occ):
        return
    while _subset_deduction(cons, val, queue):
        if not _propagate_cons(cons, val, queue, occ):
            return
    p = _branch_cell(cons, val)
    if p is None:
        # 完整解(自由格已在入口拒绝, 此处所有格均已定; 归一化去重防御)
        key = frozenset(q for q, v in val.items() if v == 1)
        if key not in seen:
            seen.add(key)
            solutions.append(dict(val))
        return
    for v in (1, 0):
        mark = dict(val)
        val[p] = v
        _search(cons, val, occ, deadline, solutions, seen, [p], limit)
        val.clear()
        val.update(mark)
        if len(solutions) >= limit:
            break


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------

def solve(n, nums, time_budget=30.0):
    """求解一道题目. 返回 (grid, info).

    grid: n×n 嵌套列表, True=涂黑. 无解抛 NoSolutionError, 超时抛
    SolverTimeoutError, 多解抛 MultiSolutionError(题目应唯一解).
    """
    sys.setrecursionlimit(max(10000, n * n * 4))
    t0 = time.time()
    _validate_input(n, nums)
    cons = build_constraints(n, nums)
    # 自由格(不在任何数字邻域内)取值不受约束 → 解必然不唯一, 直接拒绝
    covered = set()
    for _pos, _k, cells in cons:
        covered.update(cells)
    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in covered]
    if free:
        raise MultiSolutionError(
            f"{len(free)} 个格子不受任何数字约束(如 {free[0][0] + 1},"
            f"{free[0][1] + 1}), 解不唯一, 题面识别可能有误")
    val = {(r, c): -1 for r in range(n) for c in range(n)}
    occ = _cell_constraints(cons, n)
    deadline = t0 + time_budget

    solutions = []
    seen = set()
    try:
        # 初始种子: 全部数字格中心(每条约束都含自己的中心, 触发全量传播)
        _search(cons, val, occ, deadline, solutions, seen,
                [pos for pos, _k, _cells in cons], limit=2)
    except SolverTimeoutError:
        raise SolverTimeoutError(f"求解超时(>{time_budget:.0f}s, {n}x{n})")
    if not solutions:
        raise NoSolutionError("题目无解(请检查识别结果是否正确)")
    if len(solutions) >= 2:
        raise MultiSolutionError("题目存在多个合法解, 与唯一解谜题不符")
    sol = solutions[0]
    grid = [[bool(sol[(r, c)]) for c in range(n)] for r in range(n)]
    return grid, {"solve_time": time.time() - t0, "engine": "python"}


def verify(n, nums, grid):
    """独立校验一个解(与求解器逻辑完全独立). 返回 (ok, err_list)."""
    errs = []
    if len(grid) != n or any(len(row) != n for row in grid):
        errs.append(f"解的尺寸与 N={n} 不符")
        return False, errs
    for (r, c), k in sorted(nums.items()):
        cells = neighborhood(n, r, c)
        cnt = sum(1 for (rr, cc) in cells if grid[rr][cc])
        if cnt != k:
            errs.append(
                f"格({r + 1},{c + 1}) 邻域黑数 {cnt} != 数字 {k}")
    return (not errs), errs


def pretty(n, nums, grid):
    """控制台可读输出: 数字原样显示, #=黑, .=白."""
    lines = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) in nums:
                row.append(str(nums[(r, c)]))
            elif grid is not None and grid[r][c]:
                row.append("#")
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)
