# -*- coding: utf-8 -*-
"""solver.py — 帐篷与树(Tents and Trees) 纯 Python 求解器.

建模要点(双射, 见任务书 4.2):
    每顶帐篷必须与某棵树四邻接, 且树集合与帐篷集合之间存在一一配对.
    注意: 不能错误地约束"每棵树四邻接内恰好一个帐篷" —— 相邻树可能共享
    候选格, 该约束会在共享场景下误判无解. 本实现的做法:
        1) CSP 只约束"帐篷在候选格内 + 帐篷八邻不相邻 + 行列计数";
        2) 完整解再用二分图匹配(Kuhn 增广路)验证双射存在, 不成立则回溯.

对外接口:
    solve(n, trees, row_ct, col_ct, ...)     -> (grid, info) 或抛异常
    verify(n, trees, row_ct, col_ct, grid)   -> (ok, errs) 独立校验
    pretty(n, trees, row_ct, col_ct, grid)   -> 控制台可读文本

grid 为 n×n 布尔嵌套列表, True=帐篷. row_ct/col_ct: 长度 n 列表,
元素为 int 或 None(None 表示该行/列无约束).
"""
import sys
import time

# ----------------------------------------------------------------------
# 异常体系(任务书 第六步)
# ----------------------------------------------------------------------

class TentsError(Exception):
    """帐篷项目所有异常的基类."""


class NoSolutionError(TentsError):
    """题目无解(通常意味着识别有误)."""


class SolverTimeoutError(TentsError):
    """求解超出时间预算."""


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(n, r, c):
    return 0 <= r < n and 0 <= c < n


def candidate_mask(n, trees):
    """候选格掩码: 非树且至少与一棵树四邻接的格子才能放帐篷."""
    cand = [[False] * n for _ in range(n)]
    for (r, c) in trees:
        for dr, dc in NEIGH4:
            rr, cc = r + dr, c + dc
            if _in_bounds(n, rr, cc) and (rr, cc) not in trees:
                cand[rr][cc] = True
    return cand


def matching_of(n, trees, tent_cells):
    """树-帐篷二分图最大匹配(四邻接), 返回具体配对 {tree: tent}.

    完美匹配不存在时返回 None. Kuhn 算法, 规模(≤900 节点)下足够快.
    """
    tree_list = sorted(trees)
    t_idx = {t: i for i, t in enumerate(tree_list)}
    adj = [[] for _ in tree_list]
    for (r, c) in tent_cells:
        for dr, dc in NEIGH4:
            ti = t_idx.get((r + dr, c + dc))
            if ti is not None:
                adj[ti].append((r, c))
    match_tent = {}
    match_tree = [None] * len(tree_list)

    def try_kuhn(ti, visited):
        for cell in adj[ti]:
            if cell in visited:
                continue
            visited.add(cell)
            if cell not in match_tent or try_kuhn(match_tent[cell], visited):
                match_tent[cell] = ti
                match_tree[ti] = cell
                return True
        return False

    for ti in range(len(tree_list)):
        if not try_kuhn(ti, set()):
            return None
    return {tree_list[ti]: match_tree[ti] for ti in range(len(tree_list))}


# ----------------------------------------------------------------------
# 求解器: 约束传播 + 回溯
# ----------------------------------------------------------------------

class _State:
    """搜索状态. tent: -1未知 / 0无 / 1有."""

    def __init__(self, n, trees, row_ct, col_ct):
        self.n = n
        self.trees = set(trees)
        self.row_ct = list(row_ct)
        self.col_ct = list(col_ct)
        self.cand = candidate_mask(n, self.trees)
        self.tent = [[-1] * n for _ in range(n)]
        self.tree_list = sorted(self.trees)
        t_idx = {t: i for i, t in enumerate(self.tree_list)}
        self.cell_trees = {}
        self.tree_cands = [[] for _ in self.tree_list]
        for r in range(n):
            for c in range(n):
                if not self.cand[r][c]:
                    continue
                tl = []
                for dr, dc in NEIGH4:
                    ti = t_idx.get((r + dr, c + dc))
                    if ti is not None:
                        tl.append(ti)
                self.cell_trees[(r, c)] = tl
                for ti in tl:
                    self.tree_cands[ti].append((r, c))
        self.trail = []

    def set(self, r, c, v):
        """设置格值并记录撤销日志. 已有确定值时返回是否一致."""
        old = self.tent[r][c]
        if old == v:
            return True
        if old != -1:
            return False
        self.tent[r][c] = v
        self.trail.append((r, c))
        return True

    def undo_to(self, mark):
        while len(self.trail) > mark:
            r, c = self.trail.pop()
            self.tent[r][c] = -1


def _prop_line(st, k, axis, queue):
    """行(axis='row')/列(axis='col') 计数传播."""
    target = st.row_ct[k] if axis == "row" else st.col_ct[k]
    if target is None:
        return True
    n = st.n
    used = unknown = 0
    unknown_cells = []
    for i in range(n):
        r, c = (k, i) if axis == "row" else (i, k)
        v = st.tent[r][c]
        if v == 1:
            used += 1
        elif v == -1 and st.cand[r][c]:
            unknown += 1
            unknown_cells.append((r, c))
    if used > target or used + unknown < target:
        return False
    if used == target:
        for (r, c) in unknown_cells:
            if not st.set(r, c, 0):
                return False
            queue.append((r, c))
    elif used + unknown == target:
        for (r, c) in unknown_cells:
            if not st.set(r, c, 1):
                return False
            queue.append((r, c))
    return True


def _prop_tree(st, ti, queue):
    """树候选收缩: 可用候选(未知)为 0 且无相邻帐篷 → 矛盾;
    恰剩 1 个 → 强制放置. 相邻已有帐篷时不再推断(保持健全性)."""
    for (r, c) in st.tree_cands[ti]:
        if st.tent[r][c] == 1:
            return True
    avail = [(r, c) for (r, c) in st.tree_cands[ti] if st.tent[r][c] == -1]
    if not avail:
        return False
    if len(avail) == 1:
        r, c = avail[0]
        if not st.set(r, c, 1):
            return False
        queue.append((r, c))
    return True


def _propagate(st, queue):
    """增量约束传播至不动点. 返回 True=一致, False=矛盾.

    queue 中的格值刚发生变化. 规则:
      P1 帐篷八邻禁止帐篷; P2 行/列精确计数; P3 树候选收缩;
      P4 集合排除: 行/列(或某棵树)的未知候选集 U 中必然产生帐篷,
         故与 U 中所有格都八邻相邻的格子必非帐篷.
    """
    if not _propagate_basic(st, queue):
        return False
    while True:
        events = _sweep_set_exclusions(st)
        if events is False:
            return False
        ev5 = _sweep_global_count(st)
        if ev5 is False:
            return False
        events = events + ev5
        if not events:
            return True
        if not _propagate_basic(st, events):
            return False


def _propagate_basic(st, queue):
    """P1-P3 基础规则传播至不动点."""
    guard_flags = [True] * (2 * st.n)          # [rows..., cols...]
    while queue:
        r, c = queue.pop()
        v = st.tent[r][c]
        if v == 1:
            for dr, dc in NEIGH8:
                rr, cc = r + dr, c + dc
                if not _in_bounds(st.n, rr, cc):
                    continue
                nv = st.tent[rr][cc]
                if nv == 1:
                    return False              # 与已有帐篷相邻 → 矛盾
                if nv == -1:
                    if not st.set(rr, cc, 0):
                        return False
                    queue.append((rr, cc))
        if guard_flags[r]:
            guard_flags[r] = False
            if not _prop_line(st, r, "row", queue):
                return False
            guard_flags[r] = True
        if guard_flags[st.n + c]:
            guard_flags[st.n + c] = False
            if not _prop_line(st, c, "col", queue):
                return False
            guard_flags[st.n + c] = True
        for ti in st.cell_trees.get((r, c), []):
            if not _prop_tree(st, ti, queue):
                return False
    return True


def _common_neighbor_exclusion(st, cells, queue):
    """cells 全部要成为帐篷(或其中必有帐篷)时, 它们公共八邻格必非帐篷.

    返回 False=矛盾(公共邻格中已有帐篷); 否则把由此确定为"无"的格子
    作为事件加入 queue(原地修改), 返回 True.
    """
    n = st.n
    neigh = None
    for (r, c) in cells:
        s = set()
        for dr, dc in NEIGH8:
            rr, cc = r + dr, c + dc
            if _in_bounds(n, rr, cc):
                s.add((rr, cc))
        neigh = s if neigh is None else (neigh & s)
        if not neigh:
            return True
    for (r, c) in neigh:
        v = st.tent[r][c]
        if v == 1:
            return False
        if v == -1 and st.cand[r][c]:
            if not st.set(r, c, 0):
                return False
            queue.append((r, c))
    return True


def _sweep_global_count(st):
    """P5 全局计数: 双射保证 帐篷总数==树数.

    need = 树数 - 已放帐篷数; avail = 未知候选格数.
    need > avail → 矛盾; need == avail → 候选全为帐篷;
    need == 0 → 候选全为空. 返回事件列表, False=矛盾.
    """
    used = 0
    avail = 0
    avail_cells = []
    n = st.n
    for r in range(n):
        row = st.tent[r]
        cand = st.cand[r]
        for c in range(n):
            v = row[c]
            if v == 1:
                used += 1
            elif v == -1 and cand[c]:
                avail += 1
                avail_cells.append((r, c))
    need = len(st.tree_list) - used
    if need < 0 or need > avail:
        return False
    if need == avail:
        v = 1 if need > 0 else 0
        out = []
        for (r, c) in avail_cells:
            if not st.set(r, c, v):
                return False
            out.append((r, c))
        return out
    return []


def _sweep_set_exclusions(st):
    """P4 全量扫描一轮. 返回新事件列表(空=已稳定), False=矛盾."""
    n = st.n
    queue = []
    # 行/列: 未知候选集非空且大小 > 缺口(=缺口气被 P2 处理)时,
    # 公共八邻排除. 大小==缺口的情况 P2 已全置 yes.
    for k in range(n):
        target = st.row_ct[k]
        if target is not None:
            u = [(k, i) for i in range(n)
                 if st.tent[k][i] == -1 and st.cand[k][i]]
            used = sum(1 for i in range(n) if st.tent[k][i] == 1)
            if u and len(u) > target - used:
                if not _common_neighbor_exclusion(st, u, queue):
                    return False
        target = st.col_ct[k]
        if target is not None:
            u = [(i, k) for i in range(n)
                 if st.tent[i][k] == -1 and st.cand[i][k]]
            used = sum(1 for i in range(n) if st.tent[i][k] == 1)
            if u and len(u) > target - used:
                if not _common_neighbor_exclusion(st, u, queue):
                    return False
    # 树: 每棵树的未知候选集中必有它的配对帐篷(若它还没有相邻帐篷)
    for ti in range(len(st.tree_list)):
        cands = st.tree_cands[ti]
        has_tent = any(st.tent[r][c] == 1 for (r, c) in cands)
        if has_tent:
            continue
        u = [(r, c) for (r, c) in cands if st.tent[r][c] == -1]
        if not u:
            return False
        if not _common_neighbor_exclusion(st, u, queue):
            return False
    return queue


def _find_branch_cell(st):
    """按树分支: 选可用候选最少的树(≥2), 取其首个未知候选格作分支格.

    帐篷问题的分支本质是"某棵树的帐篷放哪", 围绕最受限的树展开能最快
    触发匹配/计数矛盾. 传播后可用候选==1 的树已被 P3 强制, 0 的已报矛盾.
    """
    best_tree = None
    best_avail = None
    for ti in range(len(st.tree_list)):
        avail = 0
        for (r, c) in st.tree_cands[ti]:
            if st.tent[r][c] == -1:
                avail += 1
        if avail >= 2 and (best_avail is None or avail < best_avail):
            best_avail = avail
            best_tree = ti
            if avail == 2:
                break
    if best_tree is None:
        # 兜底: 理论上不可达(不完整状态必有 ≥2 候选的树), 防御性返回
        for r in range(st.n):
            for c in range(st.n):
                if st.tent[r][c] == -1 and st.cand[r][c]:
                    return (r, c)
        return None
    for (r, c) in st.tree_cands[best_tree]:
        if st.tent[r][c] == -1:
            return (r, c)
    return None


def _is_complete(st):
    for r in range(st.n):
        row = st.tent[r]
        for c in range(st.n):
            if row[c] == -1 and st.cand[r][c]:
                return False
    return True


def _solution_grid(st):
    return [[st.tent[r][c] == 1 for c in range(st.n)] for r in range(st.n)]


def _complete_and_valid(st, counter):
    """完整解的双射验证. 合法返回 grid, 否则 None."""
    tent_cells = [(r, c) for r in range(st.n) for c in range(st.n)
                  if st.tent[r][c] == 1]
    if len(tent_cells) != len(st.tree_list):
        return None
    if matching_of(st.n, st.trees, tent_cells) is None:
        counter["bijection_fail"] += 1
        return None
    counter["found"] += 1
    return _solution_grid(st)


def _search(st, deadline, counter, limit, queue):
    """回溯搜索, 返回最多 limit 个解. queue 为刚变化的格(增量传播种子)."""
    if time.time() > deadline:
        raise SolverTimeoutError("搜索超时")
    if not _propagate(st, queue):
        return []
    if _is_complete(st):
        grid = _complete_and_valid(st, counter)
        return [grid] if grid else []
    out = []
    br, bc = _find_branch_cell(st)
    for v in (1, 0):
        mark = len(st.trail)
        if st.set(br, bc, v):
            out.extend(_search(st, deadline, counter, limit - len(out),
                               [(br, bc)]))
        st.undo_to(mark)
        if len(out) >= limit:
            break
    return out


def _initial_seed(n):
    """初始传播种子: 触发所有行与所有列."""
    seed = [(k, 0) for k in range(n)] + [(0, j) for j in range(n)]
    return list(dict.fromkeys(seed))


def solve(n, trees, row_ct, col_ct, time_budget=30.0):
    """求解一道题目. 返回 (grid, info).

    info: dict(solve_time, bijection_fail). 无解抛 NoSolutionError,
    超时抛 SolverTimeoutError.
    """
    sys.setrecursionlimit(20000)
    t0 = time.time()
    _validate_input(n, trees, row_ct, col_ct)
    st = _State(n, trees, row_ct, col_ct)
    counter = {"found": 0, "bijection_fail": 0}
    deadline = t0 + time_budget
    # 初始全量传播(处理 0 行/单候选树等开局推理与矛盾)
    queue = _initial_seed(n)
    for ti in range(len(st.tree_list)):
        if not _prop_tree(st, ti, queue):
            raise NoSolutionError("题目无解(存在无法配对的树)")
    if not _propagate(st, queue):
        raise NoSolutionError("题目无解(初始约束传播发现矛盾)")
    if _is_complete(st):
        grid = _complete_and_valid(st, counter)
        if grid is None:
            raise NoSolutionError("题目无解(帐篷-树无法一一配对)")
        return grid, {"solve_time": time.time() - t0, "bijection_fail": 0}
    try:
        sols = _search(st, deadline, counter, limit=1, queue=[])
    except SolverTimeoutError:
        raise SolverTimeoutError(f"求解超时(>{time_budget:.0f}s, {n}x{n})")
    if not sols:
        raise NoSolutionError("题目无解(请检查识别结果是否正确)")
    info = {
        "solve_time": time.time() - t0,
        "bijection_fail": counter["bijection_fail"],
    }
    return sols[0], info


# ----------------------------------------------------------------------
# 输入校验与独立校验
# ----------------------------------------------------------------------

def _validate_input(n, trees, row_ct, col_ct):
    if not isinstance(n, int) or n < 1:
        raise TentsError(f"非法网格规模 N={n}")
    if len(row_ct) != n or len(col_ct) != n:
        raise TentsError(f"行列约束长度与 N={n} 不符")
    for (r, c) in trees:
        if not _in_bounds(n, r, c):
            raise TentsError(f"树坐标越界: ({r},{c})")
    for name, arr in (("行", row_ct), ("列", col_ct)):
        for i, v in enumerate(arr):
            if v is not None and not (0 <= v <= n):
                raise TentsError(f"{name}约束越界: {name}{i + 1}={v}")
    if len(set(trees)) != len(trees):
        raise TentsError("树位置存在重复")


def verify(n, trees, row_ct, col_ct, grid):
    """独立校验一个解. 返回 (ok, err_list). 与求解器逻辑完全独立."""
    errs = []
    tset = set(trees)
    tents = []
    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                tents.append((r, c))
                if (r, c) in tset:
                    errs.append(f"格({r+1},{c+1}) 既是树又是帐篷")
    for k in range(n):
        cnt = sum(1 for c in range(n) if grid[k][c])
        if row_ct[k] is not None and cnt != row_ct[k]:
            errs.append(f"行{k+1} 帐篷数 {cnt} != 约束 {row_ct[k]}")
        cnt = sum(1 for r in range(n) if grid[r][k])
        if col_ct[k] is not None and cnt != col_ct[k]:
            errs.append(f"列{k+1} 帐篷数 {cnt} != 约束 {col_ct[k]}")
    tset2 = set(tents)
    for (r, c) in tents:
        for dr, dc in NEIGH8:
            rr, cc = r + dr, c + dc
            if (rr, cc) in tset2 and (rr, cc) > (r, c):
                errs.append(f"帐篷({r+1},{c+1})与({rr+1},{cc+1})相邻")
    if len(tents) != len(tset):
        errs.append(f"帐篷数 {len(tents)} != 树数 {len(tset)}")
    elif matching_of(n, tset, tents) is None:
        errs.append("帐篷-树不存在一一配对(双射不成立)")
    return (not errs), errs


def pretty(n, trees, row_ct, col_ct, grid):
    """控制台可读输出. T=树 A=帐篷 .=空."""
    tset = set(trees)
    head = "    " + " ".join("_" if col_ct[c] is None else str(col_ct[c])
                             for c in range(n))
    lines = [head]
    for r in range(n):
        cells = []
        for c in range(n):
            if (r, c) in tset:
                cells.append("T")
            elif grid is not None and grid[r][c]:
                cells.append("A")
            else:
                cells.append(".")
        left = "_" if row_ct[r] is None else str(row_ct[r])
        lines.append(f" {left} | " + " ".join(cells))
    return "\n".join(lines)
