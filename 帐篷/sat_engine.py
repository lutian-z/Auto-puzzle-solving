# -*- coding: utf-8 -*-
"""sat_engine.py — 帐篷与树的 CP-SAT 精确求解引擎(基于 Google OR-Tools).

引入 OR-Tools 的理由(新增依赖说明): Google 开源工业级约束求解器, 应用
广泛、API 稳定、社区成熟, Windows/Linux/macOS 均有官方 wheel。帐篷题的
"树-帐篷双射"用显式配对变量可线性精确表达, 30×30 毫秒级出解。

模型(显式匹配变量, 任务书 4.2 推荐建模方式二):
    x[c]     : 候选格 c 是否放帐篷 (候选格 = 非树且与某树四邻接).
    m[t][c]  : 树 t 的配对帐篷放在格 c (仅对 c ∈ 候选(t) 定义).
    约束:
       Σ_c m[t][c] == 1            每棵树恰配一个帐篷格;
       Σ_t m[t][c] == x[c]         帐篷格恰被一棵树认领(双射核心);
       行/列内 Σ x == 行列约束(如有);
       八邻相邻候选格对: x_a + x_b ≤ 1.
    该编码下任何可行解自动满足"每帐篷属于某树、每树恰一帐篷"的双射,
    无需事后匹配验证(独立 verify 仍作双保险)。

唯一性检查: 解出后按 x 的当前取值加阻断析取再解, 出现第二个解即多解,
INFEASIBLE 即唯一。

回退: OR-Tools 缺失/异常/超时 → solver.solve(纯 Python 传播+回溯)。
"""
import time

try:
    from ortools.sat.python import cp_model
except ImportError:                               # 环境无 OR-Tools 时可回退
    cp_model = None

import solver as _S

NEIGH8 = _S.NEIGH8
NEIGH4 = _S.NEIGH4


def _candidate_cells(n, trees):
    """候选格列表与索引(与 solver.candidate_mask 一致的定义)."""
    cand = _S.candidate_mask(n, trees)
    cells = []
    idx = {}
    for r in range(n):
        for c in range(n):
            if cand[r][c]:
                idx[(r, c)] = len(cells)
                cells.append((r, c))
    return cells, idx


def _build_model(n, trees, row_ct, col_ct):
    """构建 CP-SAT 模型. 返回 (model, x, cells, idx)."""
    m = cp_model.CpModel()
    cells, idx = _candidate_cells(n, trees)
    x = {v: m.NewBoolVar(f"x{v}") for v in range(len(cells))}
    tree_list = sorted(trees)
    # 候选格 -> 相邻树
    cell_trees = {v: [] for v in range(len(cells))}
    for ti, (r, c) in enumerate(tree_list):
        mv = []
        for dr, dc in NEIGH4:
            rr, cc = r + dr, c + dc
            v = idx.get((rr, cc))
            if v is not None:
                var = m.NewBoolVar(f"m{ti}_{v}")
                mv.append(var)
                cell_trees[v].append(var)
        if not mv:
            # 树无任何候选格 → 必无解(模型层面恒假约束直接判不可行)
            m.Add(sum([]) >= 1)
        else:
            m.Add(sum(mv) == 1)
    for v in range(len(cells)):
        cl = cell_trees[v]
        if cl:
            m.Add(sum(cl) == x[v])
        else:
            m.Add(x[v] == 0)                  # 不邻任何树的格子不能放帐篷
    # 行列计数
    row_cells = {}
    col_cells = {}
    for v, (r, c) in enumerate(cells):
        row_cells.setdefault(r, []).append(x[v])
        col_cells.setdefault(c, []).append(x[v])
    for k in range(n):
        if row_ct[k] is not None:
            m.Add(sum(row_cells.get(k, [])) == row_ct[k])
        if col_ct[k] is not None:
            m.Add(sum(col_cells.get(k, [])) == col_ct[k])
    # 八邻不相邻
    for v, (r, c) in enumerate(cells):
        for dr, dc in NEIGH8:
            w = idx.get((r + dr, c + dc))
            if w is not None and w > v:
                m.Add(x[v] + x[w] <= 1)
    return m, x, cells, idx


def _model_to_grid(cells, x, resp):
    n = 0
    for (r, _c) in cells:
        n = max(n, r + 1)
    grid = [[False] * n for _ in range(n)]
    for v, (r, c) in enumerate(cells):
        grid[r][c] = bool(resp.Value(x[v]))
    return grid


def solve_exact(n, trees, row_ct, col_ct, time_budget=20.0):
    """CP-SAT 精确求解 + 唯一性判定. 返回 (status, grid, info).

    status ∈ "ok" / "unsat" / "timeout". info["unique"]: True/False/None
    (None=预算内未确认唯一性).
    """
    if cp_model is None:
        return "timeout", None, {"reason": "no-ortools"}
    t0 = time.time()
    _S._validate_input(n, trees, row_ct, col_ct)
    model, x, cells, _idx = _build_model(n, trees, row_ct, col_ct)
    grid = None
    nsolved = 0
    unique = None
    while True:
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = max(
            0.3, time_budget - (time.time() - t0))
        sv.parameters.num_workers = 8
        st = sv.Solve(model)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if grid is None:
                grid = _model_to_grid(cells, x, sv)
            nsolved += 1
            # 阻断当前解(按 x 取值的否定析取), 找第二个解
            model.AddBoolOr([x[v].Not() if sv.Value(x[v]) else x[v]
                             for v in range(len(cells))])
            if time.time() - t0 >= time_budget:
                break                       # 来不及确认唯一性
            continue
        if st == cp_model.INFEASIBLE:
            unique = (nsolved <= 1)
        break                               # UNKNOWN/超时/已判唯一性
    info = {"elapsed": time.time() - t0, "unique": unique}
    if grid is None:
        # 首次求解即无解/超时
        if st == cp_model.INFEASIBLE:
            return "unsat", None, info
        return "timeout", None, info
    return "ok", grid, info


def solve_auto(n, trees, row_ct, col_ct, time_budget=None, log=None):
    """统一求解入口. 返回 (status, grid, info).

    70% 预算给 CP-SAT 精确模型(含唯一性判定); 不可用/超时时剩余预算
    给纯 Python 传播+回溯(solver.solve). status ∈ ok/unsat/timeout.
    """
    budget = time_budget if time_budget is not None else 60.0
    t0 = time.time()
    if cp_model is not None:
        try:
            status, grid, info = solve_exact(
                n, trees, row_ct, col_ct, time_budget=budget * 0.7)
        except Exception as e:                    # 模型异常 → 回退
            if log:
                log.info("[求解] CP-SAT 异常(%s), 转纯 Python 回退", e)
            status, grid, info = "timeout", None, {}
        if status in ("ok", "unsat"):
            info["engine"] = "cp-sat"
            info["elapsed"] = time.time() - t0
            return status, grid, info
    remain = budget - (time.time() - t0)
    if remain <= 0.5:
        return "timeout", None, {"engine": "none",
                                 "elapsed": time.time() - t0}
    try:
        grid, _sinfo = _S.solve(n, trees, row_ct, col_ct, time_budget=remain)
        info = {"engine": "python", "elapsed": time.time() - t0,
                "unique": None}
        return "ok", grid, info
    except _S.NoSolutionError:
        return "unsat", None, {"engine": "python",
                               "elapsed": time.time() - t0}
    except _S.SolverTimeoutError:
        return "timeout", None, {"engine": "python",
                                 "elapsed": time.time() - t0}
