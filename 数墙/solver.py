# -*- coding: utf-8 -*-
"""solver.py — 数墙(Nurikabe)谜题纯逻辑求解器(约束传播+回溯, 纯 Python).

不依赖任何图像/键鼠代码, 也不依赖第三方求解库; 调用方通过 _Solver 或
sat_engine.solve_auto 使用, 解一律先过 verify_shape 独立校验.

题目形式:
    shape : a×b 布尔矩阵, True=该格存在(题目区域), False=被裁掉(不存在).
            覆盖通用形式: a×b 矩形 + 四角各裁 c×d + 中间裁 e×f.
    nums  : dict {(r, c): 数字}, 数字格必定存在.
    解 grid: a×b 布尔矩阵(True=黑), 仅含存在的格子.

四条规则:
    1. 数字格必须为白;
    2. 所有黑格四向连通为一片;
    3. 不允许 2×2 全黑;
    4. 每个白色岛屿恰好含一个数字, 且面积严格等于该数字.

算法(约束传播 + 回溯):
    传播循环(每轮, P1 定岛 → P2/P3 定格 → 全局矛盾检查):
      P1 对每个数字岛(用当下状态现算, 不做增量簿记, 保证不漏解):
         - 白色洪泛(仅经白格): 得当前岛格集 ws;
         - 深度受限潜在洪泛(经所有非黑格, 距数字 ≤ 数字-1 步): 得潜在
           区域 pot(岛格到数字的白色通路长 ≤ 岛大小-1);
         矛盾: |ws|>数字 / |pot|<数字 / 岛未满但前沿为空;
         定白: |pot|==数字 → pot 全部定白; 岛未满但前沿仅 1 格 → 该格定白;
         定黑: |ws|==数字 → 岛白格的未定邻格全部定黑.
      P2(用 P1 全部改动后的新状态重算 pot)不在任何岛潜在区域的
         未定格 → 必黑.
      P3 未定格接壤两个不同岛的白色格 → 必黑; 接壤"岛白格+无主白碎片"
         时仅当碎片塞不进该岛才必黑(搜索中途同一岛可能被未定格暂时
         分成多个白色碎片, 不能当成别的岛).
      P4 出现 2×2 全黑 → 矛盾.
      P5 孤儿白区无法经非黑格到达任何其他白格 → 矛盾.
      P6 黑格潜在连通区(经所有非白格)无法连通所有黑格 → 矛盾.
      P7 (不动点后)割点式强制推断, 每应用一条即重启传播:
         岛割点: 假设岛潜在格 x 为黑后岛无法凑齐数字 → x 必白;
         海割点: 假设未定格 x 为黑后黑海潜在连通被破坏 → x 必白.
    回溯(岛屿链式分支, 空间完备且互斥):
      选缺格数最少的未完成岛, 枚举"按固定顺序首个白前沿格"(f_i 定白,
      序号更小者定黑). 岛未满必生长, 故无"就此完成"分支.
    找到的每个解都先通过独立校验 verify_shape 才记录, 保证不出错解;
    max_solutions 控制解枚举上限(用于唯一性检查).
"""
import random
import sys
import time

# 单题求解时间预算(秒), 超时返回 status="timeout"
_TIME_BUDGET = 60.0

UNKNOWN, WHITE, BLACK = 0, 1, 2
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Timeout(Exception):
    pass


class _Solver:
    def __init__(self, shape, nums, time_budget=None, max_solutions=2,
                 order_seed=0, pick_mode=0, hint=None):
        # 深盘链式分支递归较深, 放宽递归上限
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
        self.a = len(shape)
        self.b = len(shape[0]) if self.a else 0
        self.shape = [list(row) for row in shape]
        self.nums = dict(nums)
        self.time_budget = (time_budget if time_budget is not None
                            else _TIME_BUDGET)
        self.max_solutions = max_solutions
        self.order_seed = order_seed
        self.pick_mode = pick_mode
        # hint: 与 cells 同序的提示列表(WHITE/BLACK), 来自岛约束松弛 SAT
        # 的解 — 引导分支顺序(解模仿), 不参与赋值正确性, 搜索仍完备.
        self.hint = hint
        self.rng = random.Random(0x5EED ^ order_seed) if order_seed else None
        self.nodes = 0
        self.solutions = []
        self.t0 = time.time()

        for (r, c) in self.nums:
            if not (0 <= r < self.a and 0 <= c < self.b and self.shape[r][c]):
                raise ValueError(f"数字格 ({r},{c}) 不在题目区域内")

        # 格子编号与邻接表(只含存在格)
        self.idx = [[-1] * self.b for _ in range(self.a)]
        self.cells = []
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

        self.state = [UNKNOWN] * self.n
        for (r, c) in self.nums:
            self.state[self.idx[r][c]] = WHITE     # 规则1: 数字格必白
        self.num_idx = {self.idx[r][c]: v for (r, c), v in self.nums.items()}
        self.num_order = list(self.num_idx.items())   # 稳定迭代顺序
        self.num_pos = list(self.nums.keys())

    def _check_timeout(self):
        if time.time() - self.t0 > self.time_budget:
            raise Timeout()

    # 洪泛工具(列表栈实现, 热路径)
    def _flood_white(self, start):
        """仅经白格洪泛, 返回格编号列表(含 start)."""
        st = self.state
        adj = self.adj
        seen = bytearray(self.n)
        seen[start] = 1
        out = [start]
        stack = [start]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if not seen[j] and st[j] == WHITE:
                    seen[j] = 1
                    out.append(j)
                    stack.append(j)
        return out

    def _flood_pot(self, start, max_depth, blocked=None):
        """经所有非黑格的深度受限洪泛(BFS 距离 ≤ max_depth).

        返回格编号列表(含 start). blocked: 额外视为黑的格(单格假设用).
        """
        st = self.state
        if st[start] == BLACK or (blocked and start in blocked):
            return []
        adj = self.adj
        seen = bytearray(self.n)
        seen[start] = 1
        out = [start]
        frontier = [start]
        d = 0
        while frontier and d < max_depth:
            nxt = []
            for i in frontier:
                for j in adj[i]:
                    if seen[j]:
                        continue
                    jr, jc = self.cells[j]
                    if st[j] == BLACK or (blocked and j in blocked):
                        continue
                    seen[j] = 1
                    out.append(j)
                    nxt.append(j)
            frontier = nxt
            d += 1
        return out

    def _flood_nonwhite(self, start):
        """经所有非白格洪泛, 返回格编号列表."""
        st = self.state
        adj = self.adj
        seen = bytearray(self.n)
        seen[start] = 1
        out = [start]
        stack = [start]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if not seen[j] and st[j] != WHITE:
                    seen[j] = 1
                    out.append(j)
                    stack.append(j)
        return out

    def _flood_nonblack(self, start):
        """经所有非黑格洪泛(不限深度), 返回格编号列表."""
        st = self.state
        adj = self.adj
        seen = bytearray(self.n)
        seen[start] = 1
        out = [start]
        stack = [start]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if not seen[j] and st[j] != BLACK:
                    seen[j] = 1
                    out.append(j)
                    stack.append(j)
        return out

    # 传播
    def _propagate(self):
        """约束传播到不动点. 返回 (True, 岛摘要) 或 (False, None).

        岛摘要: [(num_i, target, ws列表, ws集合, frontier集合或None)],
        供回溯选岛复用(避免重复洪泛).
        """
        while True:
            changed = False
            # Phase A: 逐岛 P1 推断(用当下状态现算)
            for num_i, target in self.num_order:
                if self.state[num_i] != WHITE:
                    return (False, None)               # 规则1被破坏
                ws = self._flood_white(num_i)
                if len(ws) > target:
                    return (False, None)
                if len(ws) == target:
                    ws_set = set(ws)
                    # 岛已完成: 岛白格的未定邻格全部定黑
                    for i in ws:
                        for j in self.adj[i]:
                            if self.state[j] == UNKNOWN:
                                self.state[j] = BLACK
                                changed = True
                    continue
                pot = self._flood_pot(num_i, target - 1)
                if len(pot) < target:
                    return (False, None)
                if len(pot) == target:
                    # 潜在区域恰好够: 全部定为白
                    for j in pot:
                        if self.state[j] == UNKNOWN:
                            self.state[j] = WHITE
                            changed = True
                    continue
                # 岛未满且潜在区域有余量: 检查前沿
                frontier = []
                seenf = set()
                for i in ws:
                    for j in self.adj[i]:
                        if self.state[j] == UNKNOWN and j not in seenf:
                            seenf.add(j)
                            frontier.append(j)
                if not frontier:
                    return (False, None)               # 岛无处生长
                if len(frontier) == 1:
                    # 唯一生长点: 必白
                    self.state[frontier[0]] = WHITE
                    changed = True
            # Phase B: 用 P1 后的新状态重算 pot, 供 P2/P3
            island_whites = []
            all_potential = set()
            for num_i, target in self.num_order:
                ws = self._flood_white(num_i)
                ws_set = set(ws)
                if len(ws_set) > target:
                    return (False, None)
                pot = self._flood_pot(num_i, target - 1)
                pot_set = set(pot)
                if len(pot_set) < target:
                    return (False, None)
                island_whites.append((num_i, target, ws, ws_set, pot_set))
                all_potential |= pot_set
            # P2: 不在任何岛潜在区域的未定格必黑
            for i in range(self.n):
                if self.state[i] == UNKNOWN and i not in all_potential:
                    self.state[i] = BLACK
                    changed = True
            # P3: 接壤两个不同岛白格的未定格必黑; 接壤"岛白格+无主
            # 碎片"时仅当碎片塞不进该岛才必黑.
            for i in range(self.n):
                if self.state[i] != UNKNOWN:
                    continue
                roots = set()
                frag_cells = 0
                frag_seen = set()
                for j in self.adj[i]:
                    if self.state[j] != WHITE:
                        continue
                    r = -1
                    for (num_i, _t, ws, ws_set, _p) in island_whites:
                        if j in ws_set:
                            r = num_i
                            break
                    if r == -1:
                        if j not in frag_seen:
                            frag = self._flood_white(j)
                            frag_seen.update(frag)
                            frag_cells += len(frag)
                    else:
                        roots.add(r)
                if len(roots) > 1:
                    self.state[i] = BLACK
                    changed = True
                elif roots and frag_cells:
                    num_i = roots.pop()
                    for (ni, t, _ws, ws_set, _p) in island_whites:
                        if ni == num_i:
                            if len(ws_set) + frag_cells > t:
                                self.state[i] = BLACK
                                changed = True
                            break
            # P4: 2×2 全黑
            if self._has_2x2_black():
                return (False, None)
            # P5: 孤儿白区检查
            if not self._orphans_ok():
                return (False, None)
            # 不动点: 割点式强制推断(P7/P9) + 岛级前瞻(P10), 应用一条即重启
            if not changed:
                # P6: 黑格潜在连通(每次传播调用在此做一次)
                if not self._black_connected_potential():
                    return (False, None)
                if self._stall_deductions(island_whites):
                    changed = True
                    continue
                res = self._island_lookahead(island_whites)
                if res == "fail":
                    return (False, None)           # 存在无法补全的岛
                if res:
                    changed = True
                    continue
                return (True, island_whites)       # 真正的不动点

    def _island_completions(self, ws_set, pot_set, deficit, cap=40,
                            max_deficit=2):
        """枚举岛的全部可行最终补全(白碎片整体 all-or-nothing + 未定格).

        岛的最终形态 = ws ∪ 若干整块白色碎片 ∪ 一批未定格, 大小恰为
        缺格数且内部连通; 未并入的碎片若与最终形态相邻也必然并入(白色
        相连即同岛), 故枚举时校验"未选碎片不得与 W 相邻".
        返回 [frozenset(补全格)]; 缺格数超限、结果超过 cap 或情况复杂时
        返回 None(调用方退化为其他策略).
        """
        if deficit <= 0 or deficit > max_deficit:
            return None
        st = self.state
        cand = {i for i in pot_set if st[i] == UNKNOWN}
        if len(cand) > 24:
            return None
        # 收集与候选区相邻的白色碎片(非数字、非本岛), 整块为单位
        frag_groups = []
        seen = set()
        num_set = set(self.num_idx)
        for i in sorted(cand):
            for j in self.adj[i]:
                if st[j] != WHITE or j in ws_set or j in seen or \
                        j in num_set:
                    continue
                comp = set(self._flood_white(j))
                seen |= comp
                frag_groups.append(comp)
        units = [{i} for i in sorted(cand)] + frag_groups
        results = []
        seen_keys = set()

        def connected(W):
            start = next(iter(W))
            seen_w = {start}
            stack = [start]
            adj = self.adj
            while stack:
                i2 = stack.pop()
                for j in adj[i2]:
                    if j in W and j not in seen_w:
                        seen_w.add(j)
                        stack.append(j)
            return len(seen_w) == len(W)

        def rec(picked, W, size):
            if len(results) > cap:
                return
            if size == deficit:
                if not connected(W):
                    return
                for fg in frag_groups:
                    if fg & W:
                        continue
                    if any(any(j in W for j in self.adj[i2]) for i2 in fg):
                        return
                key = frozenset(W - ws_set)
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(key)
                return
            for k in range(len(units)):
                if picked & (1 << k):
                    continue
                u = units[k]
                if u & W:
                    continue
                if size + len(u) > deficit:
                    continue
                anchor = W if size else ws_set
                if not any(any(j in anchor for j in self.adj[c]) for c in u):
                    continue
                rec(picked | (1 << k), W | u, size + len(u))

        rec(0, set(ws_set), 0)
        if len(results) > cap:
            return None
        return results

    def _pot_contains(self, i, ws_set):
        """格 i 是否在某岛潜在区域内(近似: 经非黑可达任一 ws 格)."""
        st = self.state
        seen = bytearray(self.n)
        seen[i] = 1
        stack = [i]
        adj = self.adj
        while stack:
            cur = stack.pop()
            if cur in ws_set:
                return True
            for j in adj[cur]:
                if not seen[j] and st[j] != BLACK:
                    seen[j] = 1
                    stack.append(j)
        return False

    def _island_lookahead(self, island_whites):
        """P10 岛级前瞻: 对缺格数 ≤2 的岛枚举全部可行补全,
        所有补全共有的格 → 必白; 无一补全含有的前沿格 → 必黑.
        返回 True 表示产生新推断; 枚举发现无补全 → 矛盾返回 'fail'.
        """
        st = self.state
        for (num_i, target, ws, ws_set, pot_set) in island_whites:
            if len(ws_set) >= target:
                continue
            deficit = target - len(ws_set)
            comps = self._island_completions(ws_set, pot_set, deficit)
            if comps is None:
                continue
            if not comps:
                return "fail"                       # 岛无法补全 → 矛盾
            common = set(comps[0])
            union = set()
            for c in comps:
                common &= c
                union |= c
            for x in sorted(common):
                if st[x] == UNKNOWN:
                    st[x] = WHITE
                    return True
            # 前沿格不在任何补全中 → 必黑
            frontier = set()
            for i in ws:
                for j in self.adj[i]:
                    if st[j] == UNKNOWN:
                        frontier.add(j)
            for f in sorted(frontier - union):
                st[f] = BLACK
                return True
        return False

    def _multi_source_distance(self):
        """所有数字格为源、经非黑格的 BFS 最小步数(格编号 → 距离, -1 不可达)."""
        st = self.state
        dist = [-1] * self.n
        frontier = []
        for (r, c) in self.num_pos:
            i = self.idx[r][c]
            dist[i] = 0
            frontier.append(i)
        d = 0
        adj = self.adj
        while frontier:
            nxt = []
            for i in frontier:
                for j in adj[i]:
                    if dist[j] == -1 and st[j] != BLACK:
                        dist[j] = d + 1
                        nxt.append(j)
            frontier = nxt
            d += 1
        return dist

    def _articulation_points(self, cells_set):
        """迭代 Tarjan: 求 cells_set 导出子图的全部关节点(割点)."""
        disc = {}
        low = {}
        arts = set()
        timer = [0]
        adj = self.adj
        for root in cells_set:
            if root in disc:
                continue
            disc[root] = low[root] = timer[0]
            timer[0] += 1
            stack = [(root, -1, iter(adj[root]))]
            root_children = 0
            while stack:
                node, par, it = stack[-1]
                advanced = False
                for nb in it:
                    if nb not in cells_set or nb == par:
                        continue
                    if nb in disc:
                        if disc[nb] < low[node]:
                            low[node] = disc[nb]
                    else:
                        disc[nb] = low[nb] = timer[0]
                        timer[0] += 1
                        if node == root:
                            root_children += 1
                        stack.append((nb, node, iter(adj[nb])))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    if stack:
                        p = stack[-1][0]
                        if low[node] < low[p]:
                            low[p] = low[node]
                        if p != root and low[node] >= disc[p]:
                            arts.add(p)
            if root_children > 1:
                arts.add(root)
        return arts

    def _stall_deductions(self, island_whites):
        """不动点强制推断: P7 岛割点 / P9 海割点. 产生推断返回 True.

        P9: 黑海潜在连通图(非白格)的割点 = 染黑后会割裂黑海的精确候选集;
        P7: 岛潜在区域图的割点 ∪ 岛前沿格 = 染黑后会断岛粮道的候选集.
        候选集用一次 Tarjan 求出, 再逐格假设-洪泛校验.
        """
        st = self.state
        # P7
        for (num_i, target, ws, ws_set, pot_set) in island_whites:
            if len(ws_set) >= target:
                continue
            frontier = set()
            for i in ws:
                for j in self.adj[i]:
                    if st[j] == UNKNOWN:
                        frontier.add(j)
            cand_set = pot_set - ws_set
            arts = self._articulation_points(cand_set) if cand_set else set()
            cands = sorted(frontier | arts)
            for x in cands:
                if st[x] != UNKNOWN:
                    continue
                p2 = self._flood_pot(num_i, target - 1, {x})
                if len(p2) < target:
                    st[x] = WHITE
                    return True
        # P9
        start = None
        for i in range(self.n):
            if st[i] == BLACK:
                start = i
                break
        if start is not None:
            flood = self._flood_nonwhite(start)
            flood_set = set(flood)
            arts = self._articulation_points(flood_set)
            for x in sorted(arts):
                if st[x] != UNKNOWN:
                    continue
                saved = st[x]
                st[x] = BLACK
                ok = self._black_connected_potential()
                st[x] = saved
                if not ok:
                    st[x] = WHITE
                    return True
        return False

    def _has_2x2_black(self):
        st = self.state
        idx = self.idx
        for r in range(self.a - 1):
            row0 = idx[r]
            row1 = idx[r + 1]
            for c in range(self.b - 1):
                i00 = row0[c]
                if i00 < 0 or st[i00] != BLACK:
                    continue
                i01 = row0[c + 1]
                i10 = row1[c]
                i11 = row1[c + 1]
                if i01 >= 0 and i10 >= 0 and i11 >= 0 and \
                        st[i01] == BLACK and st[i10] == BLACK and \
                        st[i11] == BLACK:
                    return True
        return False

    def _orphans_ok(self):
        """每个白色连通块必须含数字, 或潜在可达自身之外的白格."""
        st = self.state
        num_set = set(self.num_idx)
        visited = bytearray(self.n)
        for i in range(self.n):
            if st[i] != WHITE or visited[i]:
                continue
            comp = self._flood_white(i)
            for c_ in comp:
                visited[c_] = 1
            if any(c_ in num_set for c_ in comp):
                continue
            comp_set = set(comp)
            reach = self._flood_nonblack(i)
            connected = any(st[j] == WHITE and j not in comp_set
                            for j in reach)
            if not connected:
                return False
        return True

    def _black_connected_potential(self):
        """所有黑格必须处于同一"非白潜在连通区"(规则2必要条件)."""
        st = self.state
        start = None
        for i in range(self.n):
            if st[i] == BLACK:
                start = i
                break
        if start is None:
            return True
        reach = self._flood_nonwhite(start)
        inreach = bytearray(self.n)
        for j in reach:
            inreach[j] = 1
        for i in range(self.n):
            if st[i] == BLACK and not inreach[i]:
                return False
        return True

    # 回溯
    def _search(self):
        self._check_timeout()
        self.nodes += 1
        ok, island_whites = self._propagate()
        if not ok:
            return
        st = self.state
        unknowns = [i for i in range(self.n) if st[i] == UNKNOWN]
        if not unknowns:
            grid = self._extract_grid()
            vok, _errs = verify_shape(self.shape, self.nums, grid)
            if vok:
                self.solutions.append(grid)
            return
        # 选未完成岛/分支格(复用传播摘要, 避免重复洪泛).
        # pick_mode 0: 缺格数最少(约束最强); 1: 目标数最小(岛小易推);
        # 2: 深海格二分支(黑优先) — 远离一切岛的关键未知格;
        # 3: 整岛补全分支 — 枚举缺格 ≤3 岛的全部可行最终补全, 一次定白整块;
        # 4: 扫掠序链式分支 — 按数字位置行序选岛(海矛盾早暴露);
        # 5: 扫掠序 + 整岛补全分支.
        best = None
        deep_cell = None
        deep_best = -1
        comp_island = None
        for (num_i, target, ws, ws_set, pot_set) in island_whites:
            if len(ws_set) >= target:
                continue
            frontier = set()
            for i in ws:
                for j in self.adj[i]:
                    if st[j] == UNKNOWN:
                        frontier.add(j)
            if not frontier:
                continue
            if self.pick_mode == 1:
                key = (target, target - len(ws_set))
            elif self.pick_mode in (4, 5):
                key = (num_i, target - len(ws_set))
            else:
                key = (target - len(ws_set), target)
            if best is None or key < best[0]:
                best = (key, frontier)
                comp_island = (num_i, target, ws, ws_set, pot_set)
        if self.pick_mode == 2:
            # 深海格: 距所有数字的最小 BFS 步数最大者(经非黑格)
            dist = self._multi_source_distance()
            for i in unknowns:
                if dist[i] > deep_best:
                    deep_best = dist[i]
                    deep_cell = i
        if self.pick_mode == 2 and deep_cell is not None:
            # 深海格二分支: 黑先(多数情形), 白后(一旦为白必有岛长驱直入,
            # 传播强烈级联). order_seed≥2 时白先再试.
            fcells = [deep_cell]
            if (self.hint is not None and self.hint[deep_cell] == WHITE) \
                    or (self.hint is None and self.order_seed >= 2):
                vals = ["white", "black"]
            else:
                vals = ["black", "white"]
            options = [(f, v) for f in fcells for v in vals]
            for f, val in options:
                saved = self.state[:]
                self.state[f] = WHITE if val == "white" else BLACK
                self._search()
                self.state = saved
                if len(self.solutions) >= self.max_solutions:
                    return
            return
        if self.pick_mode in (3, 5) and comp_island is not None:
            # 整岛补全分支: 每个可行补全一次定白整块(搜索深度=岛数,
            # 远小于逐格生长链); 缺格 >3 或枚举复杂时退化为链式分支.
            num_i, target, ws, ws_set, pot_set = comp_island
            deficit = target - len(ws_set)
            comps = self._island_completions(ws_set, pot_set, deficit,
                                             cap=60, max_deficit=3)
            if comps:
                if self.hint is not None:
                    comps = sorted(
                        comps,
                        key=lambda c_: sum(1 for f in c_
                                           if self.hint[f] == WHITE),
                        reverse=True)
                for opt in comps:
                    saved = self.state[:]
                    for f in opt:
                        self.state[f] = WHITE
                    self._search()
                    self.state = saved
                    if len(self.solutions) >= self.max_solutions:
                        return
                return
        if best is None:
            # 理论不可达(所有岛完成时传播会清空未定格); 兜底按黑处理
            for i in unknowns:
                st[i] = BLACK
            grid = self._extract_grid()
            vok, _errs = verify_shape(self.shape, self.nums, grid)
            if vok:
                self.solutions.append(grid)
            return
        frontier = best[1]
        # 链式分支(空间完备且互斥): "首个白前沿格" — f_i 定白,
        # 序号更小者定黑. 岛未满必生长, 故无"完成"分支.
        # 注意: 分支循环内一律通过 self.state 读写(递归会重绑 self.state),
        # 每个分支都从"当前状态的副本"出发并在返回后恢复.
        fcells = sorted(frontier)
        if self.hint is not None:
            # 提示引导(解模仿序): 提示判白的前沿格排前, 首分支即复现提示
            # 中该岛的生长方向, 搜索仅在连通性等约束迫使处偏离提示.
            fcells.sort(key=lambda f: (0 if self.hint[f] == WHITE else 1, f))
        elif self.rng is not None:
            self.rng.shuffle(fcells)     # 重启扰动: 不同尝试走不同子树序
        for ki, f in enumerate(fcells):
            saved = self.state[:]
            for kj in range(ki):
                self.state[fcells[kj]] = BLACK
            self.state[f] = WHITE
            self._search()
            self.state = saved
            if len(self.solutions) >= self.max_solutions:
                return

    def _extract_grid(self):
        st = self.state
        return [[bool(st[self.idx[r][c]] == BLACK) if self.shape[r][c]
                 else False for c in range(self.b)] for r in range(self.a)]


# 独立校验

def verify_shape(shape, nums, grid):
    """独立校验解是否满足全部四条规则. 返回 (ok, 错误列表)."""
    errors = []
    a = len(shape)
    b = len(shape[0]) if a else 0
    if grid is None:
        return False, ["解为空"]
    if len(grid) != a or any(len(row) != b for row in grid):
        return False, ["解尺寸与题目不符"]

    def exists(r, c):
        return 0 <= r < a and 0 <= c < b and shape[r][c]

    # 规则1: 数字格必须为白
    for (r, c) in nums:
        if not exists(r, c):
            errors.append(f"数字格 ({r},{c}) 不在题目区域内")
        elif grid[r][c]:
            errors.append(f"数字格 ({r},{c}) 被涂黑(规则1)")
    # 规则3: 2×2 全黑
    done = False
    for r in range(a - 1):
        for c in range(b - 1):
            if all(exists(r + dr, c + dc) and grid[r + dr][c + dc]
                   for dr in (0, 1) for dc in (0, 1)):
                errors.append(f"({r},{c}) 处出现 2×2 全黑(规则3)")
                done = True
                break
        if done:
            break
    # 规则2: 黑格连通
    blacks = [(r, c) for r in range(a) for c in range(b)
              if exists(r, c) and grid[r][c]]
    if blacks:
        seen = {blacks[0]}
        stack = [blacks[0]]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if exists(rr, cc) and grid[rr][cc] and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
        if len(seen) != len(blacks):
            errors.append(f"黑格不连通: {len(blacks)} 个黑格仅连通 "
                          f"{len(seen)} 个(规则2)")
    # 规则4: 白色岛屿恰好一数字且面积相符
    visited = [[False] * b for _ in range(a)]
    for r in range(a):
        for c in range(b):
            if not exists(r, c) or grid[r][c] or visited[r][c]:
                continue
            comp = []
            stack = [(r, c)]
            visited[r][c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in _DIRS:
                    r2, c2 = rr + dr, cc + dc
                    if exists(r2, c2) and not grid[r2][c2] and \
                            not visited[r2][c2]:
                        visited[r2][c2] = True
                        stack.append((r2, c2))
            nums_in = sorted({nums[(rr, cc)] for (rr, cc) in comp
                              if (rr, cc) in nums})
            if not nums_in:
                errors.append(f"白色岛屿({len(comp)}格, 起点({r},{c}))"
                              f"不含数字(规则4)")
            elif len(nums_in) > 1:
                errors.append(f"白色岛屿({len(comp)}格, 起点({r},{c}))"
                              f"含多个数字{nums_in}(规则4)")
            elif len(comp) != nums_in[0]:
                errors.append(f"白色岛屿({len(comp)}格, 起点({r},{c}))"
                              f"面积≠数字{nums_in[0]}(规则4)")
    return (len(errors) == 0), errors


def pretty(shape, nums, grid):
    """文本渲染题目+解: '#'=黑 '.'=白 数字=数字 ' '=不存在格(两位宽对齐)."""
    a = len(shape)
    b = len(shape[0]) if a else 0
    lines = []
    for r in range(a):
        row = []
        for c in range(b):
            if not shape[r][c]:
                row.append("  ")
            elif (r, c) in nums:
                row.append(f"{nums[(r, c)]:<2d}")
            elif grid is None:
                row.append(" ?")
            else:
                row.append(" #" if grid[r][c] else " .")
        lines.append("".join(row))
    return "\n".join(lines)


def shape_params(shape):
    """把存在矩阵参数化为通用形式 (a, b, corner, middle).

    corner: [(行数,列数)×4] 对应左上/右上/左下/右下, (0,0) 表示未裁;
    middle: (行起点,列起点,行数,列数) 或 None.
    不符合通用形式返回 None.
    """
    a = len(shape)
    b = len(shape[0]) if a else 0
    miss = [(r, c) for r in range(a) for c in range(b) if not shape[r][c]]
    if not miss:
        return a, b, [(0, 0)] * 4, None
    miss_set = set(miss)

    def corner_size(r0, c0, dr, dc):
        """从角 (r0,c0) 向内量被裁矩形的 (行数,列数), 无裁返回 (0,0)."""
        rows = 0
        while rows < a and (r0 + dr * rows, c0) in miss_set:
            rows += 1
        if rows == 0:
            return (0, 0)
        cols = 0
        while cols < b and (r0, c0 + dc * cols) in miss_set:
            cols += 1
        for r in range(rows):
            for c in range(cols):
                if (r0 + dr * r, c0 + dc * c) not in miss_set:
                    return None
        return (rows, cols)

    tl = corner_size(0, 0, 1, 1)
    tr = corner_size(0, b - 1, 1, -1)
    bl = corner_size(a - 1, 0, -1, 1)
    br = corner_size(a - 1, b - 1, -1, -1)
    if None in (tl, tr, bl, br):
        return None
    for (r0, c0, dr, dc), (rows, cols) in zip(
            [(0, 0, 1, 1), (0, b - 1, 1, -1), (a - 1, 0, -1, 1),
             (a - 1, b - 1, -1, -1)], [tl, tr, bl, br]):
        for r in range(rows):
            for c in range(cols):
                miss_set.discard((r0 + dr * r, c0 + dc * c))
    if not miss_set:
        return a, b, [tl, tr, bl, br], None
    rs = [r for r, _ in miss_set]
    cs = [c for _, c in miss_set]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if len(miss_set) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return a, b, [tl, tr, bl, br], (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
