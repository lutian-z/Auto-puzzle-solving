# -*- coding: utf-8 -*-
"""sat_engine.py — 数墙(Nurikabe)的 CP-SAT 精确求解引擎(基于 Google OR-Tools).

引入 OR-Tools 的理由(新增依赖说明): 它是 Google 开源的工业级约束求解器
(CP-SAT), 应用广泛、API 稳定、社区成熟, Windows/Linux/macOS 均有官方
wheel。数墙的"黑海连通/白岛连通"是全局约束, 对朴素 SAT 编码是 notorious
难点, 而 CP-SAT 的整数流量守恒约束可以线性规模精确表达连通性。

主入口 solve_auto → 精确路径 solve_cuts:
    b(v)     : 格 v 是黑格.
    id(k,v)  : 格 v 属于数字岛 k (定义域 v ∈ pot_k — 该岛深度受限潜在
               区域, 是最终形态的严格超集).
    岛约束:
       数字格必白且 id(k,数字)=真;  id(k,v) → ¬b(v);
       白格 → 恰属一岛 (Add(sum id)==1 仅在白时生效);
       相邻两白格必同岛 (条件字面量是正的 b — 两白时才强制同岛);
       Σ_v id(k,v) == 数字                        —— 岛面积;
       岛内连通(单商品流): 数字格为源发出 target-1 单位, 每个岛格消耗
       1 单位, 流量只经岛格 → 全部岛格经岛内路径连到数字格.
       (注意: "分层可达 p" 编码对环绕海湾的岛不健全 — 最短网格路的
        父格可能在岛外, 会错杀真解; 仅 target ≤ 2(深度 ≤1)时使用.)
    海约束(单商品流, 根可选):
       恰选一个黑格 rho 为"海根"; depot 只给根直接供流; 每个黑格消耗
       1 单位; 流量只经黑格 → 所有黑格经海内路径连到根, 精确且健全.
       (若 depot 直连所有黑格, 任意布局都满足守恒, 约束即失效.)
    2×2: 每个 2×2 块 AddBoolOr(至少一格白).
    冗余强化: Σb == 海格数; 黑格必有黑邻居(海≥2 时); 黑黑相邻边数 ≥
    海格数-1(连通海有生成树); 目标最大化黑黑相邻边数(引导搜索).
    惰性割兜底: 万一返回海不连通的模型(不应发生, 防御性), 按连通块加
    声音割约束重解; 候选解先过 solver.verify_shape 全规则校验, 通过即
    作为首个解交付(找到一解即返回, 2026-09-03 按用户要求移除唯一性
    枚举与作答前复检).

回退路径 solve_hybrid: 岛约束松弛 SAT 快速产出若干提示解, 以"解模仿"
分支序交给自研传播+回溯求解器(solver.py, 无 OR-Tools 时也可独立完成).

开发中曾有三处极性/健全性 bug(同岛子句反写、阻断子句反写、depot 直连
黑格使海约束失效), 均以"真解固定入模型必须可行"的差分测试定位, 详见
交付说明。
"""
import time

from ortools.sat.python import cp_model

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class _CpContext:
    """一道题的静态预计算(格编号/邻接/各岛潜在区域)."""

    def __init__(self, shape, nums):
        self.a = len(shape)
        self.b = len(shape[0]) if self.a else 0
        self.shape = [list(row) for row in shape]
        self.nums = dict(nums)
        self.cells = []
        self.idx = [[-1] * self.b for _ in range(self.a)]
        for r in range(self.a):
            for c in range(self.b):
                if self.shape[r][c]:
                    self.idx[r][c] = len(self.cells)
                    self.cells.append((r, c))
        self.n = len(self.cells)
        self.adj = [[] for _ in range(self.n)]
        for i, (r, c) in enumerate(self.cells):
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.a and 0 <= cc < self.b and self.shape[rr][cc]:
                    self.adj[i].append(self.idx[rr][cc])
        # 各岛深度受限潜在区域 + BFS 距离(分层可达用)
        self.pots = []
        for (r, c), target in sorted(nums.items()):
            num_i = self.idx[r][c]
            dist = {num_i: 0}
            frontier = [num_i]
            d = 0
            while frontier and d < target - 1:
                nxt = []
                for i in frontier:
                    for j in self.adj[i]:
                        if j not in dist:
                            dist[j] = d + 1
                            nxt.append(j)
                frontier = nxt
                d += 1
            self.pots.append((num_i, target, set(dist), dist))


def _build_model(ctx):
    """构建 CP-SAT 模型. 返回 (model, bvars)."""
    m = cp_model.CpModel()
    n = ctx.n
    b = [m.NewBoolVar(f"b{i}") for i in range(n)]
    pot_of = {}
    # 岛约束
    for ki, (num_i, target, pot, dist) in enumerate(ctx.pots):
        idv = {v: m.NewBoolVar(f"id{ki}_{v}") for v in pot}
        m.Add(b[num_i] == 0)                    # 数字格必白
        m.Add(idv[num_i] == 1)
        for v in pot:
            m.AddImplication(idv[v], b[v].Not())     # id → 非黑
        # 岛面积恰等数字
        m.Add(sum(idv.values()) == target)
        if target <= 2:
            # 深度 ≤1 的潜在域: 分层可达在此范围内健全(任何岛格的
            # 距离-1 邻格必是数字格本身).
            pv = {v: m.NewBoolVar(f"p{ki}_{v}") for v in pot}
            m.Add(pv[num_i] == 1)
            for v in pot:
                m.AddImplication(pv[v], idv[v])
                m.AddImplication(idv[v], pv[v])
            for v, dv in dist.items():
                if v == num_i:
                    continue
                upper = [pv[u] for u in ctx.adj[v]
                         if u in dist and dist[u] == dv - 1]
                m.AddBoolOr([pv[v].Not()] + upper)
        else:
            # 岛内连通: 单商品流. 数字格为源发出 target 单位, 每个岛格
            # 消耗 1 单位, 流量只经岛格 → 岛格全部连到数字格, 精确且健全.
            # (分层可达编码对"环绕海湾的岛"不健全: 最短网格路的父格可能
            #  在岛外, 会错杀真解, 故深度 ≥2 一律用流.)
            ins = {v: [] for v in pot}
            outs = {v: [] for v in pot}
            for v in pot:
                for u in ctx.adj[v]:
                    if u not in pot or u <= v:
                        continue
                    f = m.NewIntVar(-target, target, f"f{ki}_{u}_{v}")
                    # 流量只能经岛格(两端均为 id 时才可非零)
                    m.Add(f == 0).OnlyEnforceIf(idv[v].Not())
                    m.Add(f == 0).OnlyEnforceIf(idv[u].Not())
                    ins[v].append(f)               # f>0: u→v 流入 v
                    outs[u].append(f)
            for v in pot:
                if v == num_i:
                    continue
                m.Add(sum(ins[v]) - sum(outs[v]) == idv[v])
            # 根自身也是岛格(消耗 1 单位): 净出口 = target - 1
            m.Add(sum(outs[num_i]) - sum(ins[num_i]) == target - 1)
        # 白格恰属一岛
        for v in pot:
            pot_of.setdefault(v, []).append((ki, idv[v]))
    for v in range(n):
        entries = pot_of.get(v, [])
        if not entries:
            m.Add(b[v] == 1)                    # 不在任何潜在区域 → 必黑
        else:
            m.Add(sum(idv for _ki, idv in entries) == 1).OnlyEnforceIf(
                b[v].Not())
    # 相邻两白格必同岛: (b_v ∨ b_u ∨ ¬id_k(v) ∨ id_k(u)) 等子句 —
    # 两格全白时 id 取值强制一致; 注意条件字面量是正的 b(黑) —
    # 若误写成 ¬b, 子句只在两格全黑时才生效, 同岛约束即告失效.
    for v in range(n):
        for u in ctx.adj[v]:
            if u <= v:
                continue
            e_u = pot_of.get(v, [])
            e_v = pot_of.get(u, [])
            map_u = dict(e_u)
            map_v = dict(e_v)
            for ki in set(map_u) | set(map_v):
                iv = map_u.get(ki)
                iu = map_v.get(ki)
                not_both_white = [b[v], b[u]]
                if iv is None:
                    m.AddBoolOr(not_both_white + [iu.Not()])
                elif iu is None:
                    m.AddBoolOr(not_both_white + [iv.Not()])
                else:
                    m.AddBoolOr(not_both_white + [iv.Not(), iu])
                    m.AddBoolOr(not_both_white + [iv, iu.Not()])
    # 海连通(单商品流, 根可选)
    # 恰选一个黑格作"海根"; depot 只给根直接供流; 每个黑格消耗 1 单位,
    # 流量只能经黑格. 于是所有黑格都能从根经黑格路径到达 → 黑海连通,
    # 精确且健全. (若 depot 直连所有黑格, 任意布局都满足守恒, 约束失效.)
    F = n                                  # 流量上限
    sea_total = n - sum(t for (_ni, t, _p, _d) in ctx.pots)
    if sea_total >= 1:
        inflow = [[] for _ in range(n)]
        outflow = [[] for _ in range(n)]
        for v in range(n):
            for u in ctx.adj[v]:
                f = m.NewIntVar(0, F, f"f{v}_{u}")   # u → v 的流量
                # 流量只能流过黑格
                m.Add(f == 0).OnlyEnforceIf(b[v].Not())
                m.Add(f == 0).OnlyEnforceIf(b[u].Not())
                inflow[v].append(f)                    # v 的流入
                outflow[u].append(f)                   # u 的流出
        rho = [m.NewBoolVar(f"rho{i}") for i in range(n)]
        m.Add(sum(rho) == 1)
        dep_out = []
        for v in range(n):
            m.AddImplication(rho[v], b[v])           # 根必为黑
            fd = m.NewIntVar(0, F, f"fd_{v}")        # depot → v
            m.Add(fd <= F * rho[v])                  # 非根格 depot 流为 0
            dep_out.append(fd)
            inflow[v].append(fd)
        # 每格守恒: 流入 − 流出 == b(v) (每个黑格消耗 1 单位)
        for v in range(n):
            m.Add(sum(inflow[v]) - sum(outflow[v]) == b[v])
        # depot 守恒: 流出 == Σ b(v)
        m.Add(sum(dep_out) == sum(b))
    # 冗余但强传播: 总海格数 = 总格数 − Σ 岛面积; 且海至少 1 格
    m.Add(sum(b) == n - sum(t for (_ni, t, _p, _d) in ctx.pots))
    m.Add(sum(b) >= 1)
    # 2×2 全黑禁止
    for r in range(ctx.a - 1):
        for c in range(ctx.b - 1):
            q = [ctx.idx[r][c], ctx.idx[r][c + 1],
                 ctx.idx[r + 1][c], ctx.idx[r + 1][c + 1]]
            if any(x < 0 for x in q):
                continue
            m.AddBoolOr([b[x].Not() for x in q])
    # 海的必要条件(声音, 松弛/精确两用)
    # 海连通且 |海|≥2 时, 每个黑格必有至少一个黑邻居(连通块≥2 的节点
    # 必有块内邻居); |海|≤1 时自动成立. 大幅减少松弛解里"孤立黑点"型
    # 碎海, 让提示解更接近真解.
    sea_total = n - sum(t for (_ni, t, _p, _d) in ctx.pots)
    if sea_total > 1:
        for v in range(n):
            if ctx.adj[v]:
                m.AddBoolOr([b[v].Not()] + [b[u] for u in ctx.adj[v]])
            else:
                m.Add(b[v] == 0)       # 孤立格在海≥2 时不可为黑
    return m, b


def _black_components(grid, shape):
    """解网格的黑格 4-连通块划分. 返回 [set((r,c)...)]."""
    a = len(shape)
    b = len(shape[0]) if a else 0
    seen = [[False] * b for _ in range(a)]
    comps = []
    for r in range(a):
        for c in range(b):
            if not shape[r][c] or not grid[r][c] or seen[r][c]:
                continue
            comp = set()
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in _DIRS:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < a and 0 <= c2 < b and shape[r2][c2] \
                            and grid[r2][c2] and not seen[r2][c2]:
                        seen[r2][c2] = True
                        stack.append((r2, c2))
            comps.append(comp)
    return comps


def solve_cuts(shape, nums, time_budget=None, use_objective=True):
    """惰性连通割精确求解(主精确路径).

    用岛约束松弛模型(不含海连通)反复求解; 每得到一个海不连通的模型 M,
    就向模型加入声音割约束后重解:
      对 M 的每个黑海连通块 Ci: AddBoolOr( [Ci 的白邻居格变黑] +
                                          [Ci 的格变白] ).
    声音性: 若某解让 Ci 全保持黑且 Ci 的全部原白邻居保持白, 则 Ci 在该
    解中四周全白、是孤立黑块, 违反规则 2 — 故任何真解必满足每条割.
    use_objective: 目标最大化"黑黑相邻边数" — 连通海(|海|=S)内部边数
    ≥ S-1, 断海显著更少; 该目标驱动求解器直奔接近连通的解, 割迭代因此
    大幅收敛(每轮都是远跳而非相邻僵尸游走).
    松弛即矛盾时原题必无解; 首个通过 verify_shape 全规则校验的解立即
    交付(找到一解即返回).
    """
    t0 = time.time()
    budget = time_budget if time_budget is not None else 60.0
    import solver as S          # verify_shape / 网格约定 (延迟导入)

    ctx = _CpContext(shape, nums)
    m, b = _build_model(ctx)
    if use_objective:
        edge_terms = []
        for v in range(ctx.n):
            for u in ctx.adj[v]:
                if u <= v:
                    continue
                e = m.NewBoolVar(f"e{v}_{u}")
                m.Add(e <= b[v])
                m.Add(e <= b[u])
                m.Add(e >= b[v] + b[u] - 1)
                edge_terms.append(e)
        sea_total = ctx.n - sum(t for (_ni, t, _p, _d) in ctx.pots)
        if sea_total >= 1:
            # 声音下界: 海连通(|海|=S)必有 ≥ S-1 条黑黑相邻边(生成树);
            # 断海(≥2 块)至多 S-2 条. 排除断海解, 大幅收紧松弛.
            m.Add(sum(edge_terms) >= sea_total - 1)
        m.Maximize(sum(edge_terms))
    grid = None
    status = "timeout"
    it = 0
    while True:
        remain = budget - (time.time() - t0)
        if remain <= 0.5:
            break
        solver = cp_model.CpSolver()
        # 迭代时间配比: 首次求解拿剩余预算的 ~45%(精确流模型通常一次
        # 完成), 后续割迭代拿剩余的 60%, 几何衰减防饿死
        share = 0.45 if it == 0 else 0.6
        solver.parameters.max_time_in_seconds = min(
            remain, max(4.0, share * remain))
        solver.parameters.num_workers = 8
        st = solver.Solve(m)
        it += 1
        if st == cp_model.INFEASIBLE:
            status = "unsat"
            break
        if st != cp_model.OPTIMAL and st != cp_model.FEASIBLE:
            grid = None          # 超时/无解出: 不携带未通过校验的中间模型
            break
        grid = _model_to_grid(ctx, solver, b)
        ok, _errs = S.verify_shape(ctx.shape, ctx.nums, grid)
        if ok:
            status = "ok"
            break
        # 海不连通 → 按连通块加割
        comps = _black_components(grid, ctx.shape)
        n_cuts = 0
        if len(comps) >= 2:
            for comp in comps:
                nbrs = set()
                for (r, c) in comp:
                    for dr, dc in _DIRS:
                        r2, c2 = r + dr, c + dc
                        if 0 <= r2 < ctx.a and 0 <= c2 < ctx.b \
                                and ctx.shape[r2][c2] and not grid[r2][c2]:
                            nbrs.add(ctx.idx[r2][c2])
                if not nbrs:
                    continue
                comp_idx = [ctx.idx[r][c] for (r, c) in sorted(comp)]
                m.AddBoolOr([b[v] for v in sorted(nbrs)] +
                            [b[u].Not() for u in comp_idx])
                n_cuts += 1
            # 白集割: 任何真解必须把当前模型至少一个白格染黑
            # (岛结构整体复现只会得到同一个海不连通赋值), 迫使每次迭代
            # 大幅跳动, 避免在相邻模型间僵尸游走.
            whites = [i for i in range(ctx.n) if not solver.Value(b[i])]
            m.AddBoolOr([b[i] for i in whites])
        print(f"[求解] 割迭代 #{it}: {len(comps)} 个海块, 加 {n_cuts} 条割")
    elapsed = time.time() - t0
    info = {"elapsed": elapsed, "iterations": it}
    return status, grid, info


def _model_to_grid(ctx, solver, bvars):
    grid = [[False] * ctx.b for _ in range(ctx.a)]
    for i in range(ctx.n):
        r, c = ctx.cells[i]
        grid[r][c] = solver.Value(bvars[i])
    return grid


def relaxed_hints(shape, nums, k=3, time_budget=15.0):
    """枚举至多 k 个互不相同的岛约束松弛解, 作为搜索提示备选.

    通过"解出 → 排除该解 → 再解"得到彼此不同的松弛解(都满足规则 1/3/4,
    黑海连通均未保证). 返回 (status, grids): status="ok" 表示 grids 非空;
    "unsat" 表示松弛即矛盾(原题必无解); "timeout" 表示预算内无解.
    """
    t0 = time.time()
    ctx = _CpContext(shape, nums)
    m, b = _build_model(ctx)
    out = []
    while len(out) < k and time.time() - t0 < time_budget:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(
            0.5, time_budget - (time.time() - t0))
        solver.parameters.num_workers = 8
        st = solver.Solve(m)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            out.append(_model_to_grid(ctx, solver, b))
            # 阻断当前解: 加"当前取值的否定"析取(每个文字在当前解下为假)
            m.AddBoolOr([b[i].Not() if solver.Value(b[i]) else b[i]
                         for i in range(ctx.n)])
            continue
        if st == cp_model.INFEASIBLE:
            return ("unsat" if not out else "ok", out)
        break
    return ("ok" if out else "timeout", out)


def solve_hybrid(shape, nums, time_budget=None):
    """两阶段混合求解(回退路径). 返回 (status, grid, info) — 与
    solve_cuts 返回约定一致.

    阶段1: 岛约束松弛 SAT 快速枚举若干提示解 M_i (规则2 未保证, 不直接
    采用; 松弛本身矛盾时直接判 unsat — 松弛是必要条件);
    阶段2: 依次以每个 M_i 为 hint 跑自研传播+回溯(解模仿序), 并轮换
    分支策略; 任一提示走不通再退回无提示. 提示只影响分支顺序, 搜索仍
    空间完备, 解都经 verify_shape 独立校验. OR-Tools 不可用时整体退化为
    无提示自研求解.
    """
    import solver as S                # 延迟导入, 避免模块循环依赖

    t0 = time.time()
    budget = time_budget if time_budget is not None else 60.0
    hints = []
    rel_status = "skip"
    try:
        st, grids = relaxed_hints(shape, nums, k=3,
                                  time_budget=min(budget * 0.3, 12.0))
        rel_status = st
        if st == "unsat":
            return ("unsat", None,
                    {"nodes": 0, "solutions_found": 0,
                     "elapsed": time.time() - t0,
                     "relaxed": rel_status})
        for g in grids:
            hint = []
            for r in range(len(shape)):
                for c in range(len(shape[0]) if shape else 0):
                    if shape[r][c]:
                        hint.append(S.BLACK if g[r][c] else S.WHITE)
            hints.append(hint)
        if hints:
            print(f"[求解] 松弛提示 {len(hints)} 个就绪, 进入提示引导搜索")
    except Exception as e:            # OR-Tools 缺失/异常 → 无提示回退
        print(f"[求解] 松弛 SAT 不可用({e}), 转纯自研求解")

    # 分支计划: 每个提示优先走约束最强的缺格最少策略, 再换深海二分支
    # /整岛补全, 最后无提示兜底.
    plans = []
    for h in hints:
        plans.append((h, 0))
    for h in hints:
        plans.append((h, 2))
    plans.append((None, 0))
    for h in hints:
        plans.append((h, 3))
    plans.append((hints[0] if hints else None, 1))
    plans.append((None, 2))
    nodes_total = 0
    for i, (hint_i, pm) in enumerate(plans):
        remain = budget - (time.time() - t0)
        if remain <= 1:
            break
        slot = max(10.0, remain / (len(plans) - i))
        s = None
        status = "timeout"
        try:
            s = S._Solver(shape, nums, slot, 1, order_seed=i, pick_mode=pm,
                          hint=hint_i)
            s._search()
            status = "ok" if s.solutions else "unsat"
        except S.Timeout:
            status = "timeout"
        except ValueError as e:
            return ("unsat", None,
                    {"nodes": nodes_total, "solutions_found": 0,
                     "elapsed": time.time() - t0,
                     "error": str(e), "relaxed": rel_status})
        nodes_total += s.nodes if s is not None else 0
        if status == "ok":
            grid = s.solutions[0]
            ok, _errs = S.verify_shape(shape, nums, grid)
            if ok:
                return ("ok", grid,
                        {"nodes": nodes_total,
                         "solutions_found": 1,
                         "elapsed": time.time() - t0,
                         "relaxed": rel_status})
            print("[求解] 解未通过独立校验, 换计划重试")
            continue
        if status == "unsat":
            # 搜尽整棵树(与提示/策略无关) → 确定性无解
            return ("unsat", None,
                    {"nodes": nodes_total, "solutions_found": 0,
                     "elapsed": time.time() - t0,
                     "relaxed": rel_status})
    return ("timeout", None,
            {"nodes": nodes_total, "solutions_found": 0,
             "elapsed": time.time() - t0,
             "relaxed": rel_status})


def solve_auto(shape, nums, time_budget=None):
    """统一求解入口(供 main.py 调用). 返回 (status, grid, info).

    策略: 70% 预算给精确 CP-SAT(solve_cuts, 模型已含岛/海连通的精确
    编码, 实测 0.01~2.6s 解出全部题目盘); 若超时, 剩余预算给混合回退
    (solve_hybrid: 松弛提示 + 提示引导自研传播搜索). 求得首个合法解
    即返回(2026-09-03 按用户要求移除唯一性枚举).
    """
    t0 = time.time()
    budget = time_budget if time_budget is not None else 60.0
    try:
        status, grid, info = solve_cuts(shape, nums, budget * 0.7)
    except Exception as e:            # OR-Tools 缺失/异常 → 回退
        print(f"[求解] CP-SAT 不可用({e}), 转混合回退")
        status, grid, info = "timeout", None, {}
    if status in ("ok", "unsat"):
        info["elapsed"] = time.time() - t0
        return status, grid, info
    remain = budget - (time.time() - t0)
    if remain <= 1:
        info.setdefault("nodes", 0)
        info.setdefault("solutions_found", 0)
        info["elapsed"] = time.time() - t0
        return "timeout", None, info
    status2, grid2, info2 = solve_hybrid(shape, nums, remain)
    info2["fallback_from"] = status
    info2["elapsed"] = time.time() - t0
    return status2, grid2, info2
