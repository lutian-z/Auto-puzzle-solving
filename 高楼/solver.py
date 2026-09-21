# -*- coding: utf-8 -*-
"""solver.py — 高楼谜题(带可见性约束的拉丁方)纯逻辑求解器.

不依赖任何图像/键鼠代码, 可单独测试. 支持 n=4..9, 四周约束缺失用 0 表示,
宫内初值用 given 矩阵(0 表示空格)表示.

约定:
    top[j] / bottom[j] : 第 j 列的上/下约束(0=无约束)
    left[i] / right[i] : 第 i 行的左/右约束(0=无约束)
    given[r][c]        : 宫内初值(0=空)

算法(AC-3 弧一致 + MRV 回溯):
    1. 对每个 (左约束,右约束) 组合预生成满足两侧可见性约束的全部排列并缓存;
    2. 每条有约束的行/列维护"可行排列集合"及每个位置的可能取值(弧一致);
    3. 传播: 若某格候选只剩一个值, 或某行/列在某个位置只剩一个可能值, 强制填入;
    4. 回溯: 候选数最少的格子(MRV)尝试取值. 可解稀疏线索的 8x8/9x9.
"""
import itertools
import time

import numpy as np

_TIME_BUDGET = 20.0   # 单题求解时间预算(秒), 超时返回 None
TIGHT_MAX = 10 ** 9   # 有约束的行/列全部使用弧一致传播(联合行↔列过滤)


# ----------------------------------------------------------------------
# 可见性
# ----------------------------------------------------------------------

def visible_count(seq):
    """从左侧看 seq 能看到的高楼数."""
    mx = 0
    cnt = 0
    for v in seq:
        if v > mx:
            mx = v
            cnt += 1
    return cnt


# ----------------------------------------------------------------------
# 排列缓存
# ----------------------------------------------------------------------

_PERM_CACHE = {}
_ROW_IDX_CACHE = {}


def _perms(n):
    """返回 P: 1..n 全部排列矩阵 (n! × n, int8). 缓存."""
    if n not in _PERM_CACHE:
        import math
        it = itertools.chain.from_iterable(
            itertools.permutations(range(1, n + 1)))
        P = np.fromiter(it, dtype=np.int8,
                        count=n * math.factorial(n)).reshape(-1, n)
        _PERM_CACHE[n] = P
    return _PERM_CACHE[n]


def _row_feasible_idx(n, c_left, c_right):
    """返回满足"左可见数=c_left 且 右可见数=c_right"的全部排列下标数组.

    约束为 0 表示该侧不限. 结果缓存, 跨题复用. 无约束返回 None.
    """
    if c_left == 0 and c_right == 0:
        return None
    key = (n, c_left, c_right)
    hit = _ROW_IDX_CACHE.get(key)
    if hit is not None:
        return hit
    P = _perms(n)
    m = P.shape[0]
    # 可见数 = 前缀严格新高的个数; 用累积最大值一次向量化算出
    def _vis_counts(Q):
        cm = np.maximum.accumulate(Q, axis=1)
        cnt = np.ones(m, dtype=np.int16)          # 首位必可见
        if Q.shape[1] > 1:
            cnt += (Q[:, 1:] > cm[:, :-1]).sum(axis=1)
        return cnt
    lv = _vis_counts(P)
    rv = _vis_counts(P[:, ::-1])
    if c_left == 0:
        idx = np.where(rv == c_right)[0]
    elif c_right == 0:
        idx = np.where(lv == c_left)[0]
    else:
        idx = np.where((lv == c_left) & (rv == c_right))[0]
    _ROW_IDX_CACHE[key] = idx
    return idx


# ----------------------------------------------------------------------
# AC-3 求解器
# ----------------------------------------------------------------------

class Inconsistent(Exception):
    pass


class _Solver:
    def __init__(self, n, top, bottom, left, right, given):
        self.n = n
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.P = _perms(n)
        self.given = [list(row) for row in given]
        self.grid = [list(row) for row in given]
        self.row_used = [0] * n
        self.col_used = [0] * n
        self.trail = []
        self.nodes = 0
        self._t0 = time.time()

        self.row_cur = []
        self.row_possible = []
        self.col_cur = []
        self.col_possible = []
        for i in range(n):
            idx = _row_feasible_idx(n, left[i], right[i])
            if idx is None or len(idx) > TIGHT_MAX:
                # 无约束或可行排列过多 -> 松散: 仅拉丁占用 + 完成时校验线索
                self.row_cur.append(None)
                self.row_possible.append(None)
            else:
                if len(idx) == 0:
                    raise Inconsistent(f"行{i+1} 的约束组合无可行排列")
                self.row_cur.append(idx.copy())
                self.row_possible.append(None)
        for j in range(n):
            idx = _row_feasible_idx(n, top[j], bottom[j])
            if idx is None or len(idx) > TIGHT_MAX:
                self.col_cur.append(None)
                self.col_possible.append(None)
            else:
                if len(idx) == 0:
                    raise Inconsistent(f"列{j+1} 的约束组合无可行排列")
                self.col_cur.append(idx.copy())
                self.col_possible.append(None)

        # 初值占用与唯一性
        for r in range(n):
            for c in range(n):
                v = self.grid[r][c]
                if v:
                    if (self.row_used[r] >> v) & 1 or (self.col_used[c] >> v) & 1:
                        raise Inconsistent(f"初值在行/列重复: ({r+1},{c+1})={v}")
                    self.row_used[r] |= 1 << v
                    self.col_used[c] |= 1 << v
        # 初值过滤行/列可行排列集
        for r in range(n):
            for c in range(n):
                v = self.grid[r][c]
                if v:
                    if not self._apply_filter("row", r, c, v) or \
                       not self._apply_filter("col", c, r, v):
                        raise Inconsistent(f"初值与行/列约束矛盾: ({r+1},{c+1})={v}")
        # 初始化每线的每位置可能值(为 None 的无约束行/列保持 None)
        for i in range(n):
            if self.row_cur[i] is not None:
                self._recompute_possible("row", i)
        for j in range(n):
            if self.col_cur[j] is not None:
                self._recompute_possible("col", j)

    # ---------------- 基础操作 ----------------
    def _recompute_possible(self, kind, idx_):
        """重新计算一条线的每位置可能取值(整数位掩码). kind: 'row'/'col'."""
        n = self.n
        cur = self.row_cur[idx_] if kind == "row" else self.col_cur[idx_]
        if cur is None:
            return None
        sub = self.P[cur]
        poss = []
        for p in range(n):
            cnts = np.bincount(sub[:, p], minlength=n + 1)
            mask = 0
            for v in range(1, n + 1):
                if cnts[v]:
                    mask |= 1 << v
            poss.append(mask)
        if kind == "row":
            self.row_possible[idx_] = poss
        else:
            self.col_possible[idx_] = poss

    def _apply_filter(self, kind, idx_, pos, val):
        """把线 idx_ 的可行排列过滤为"位置 pos = val". 空集返回 False."""
        if kind == "row":
            cur = self.row_cur[idx_]
            if cur is None:
                return True
            sub = self.P[cur]
            new = cur[sub[:, pos] == val]
            self.row_cur[idx_] = new
            if len(new) == 0:
                return False
            self._recompute_possible("row", idx_)
            return True
        else:
            cur = self.col_cur[idx_]
            if cur is None:
                return True
            sub = self.P[cur]
            new = cur[sub[:, pos] == val]
            self.col_cur[idx_] = new
            if len(new) == 0:
                return False
            self._recompute_possible("col", idx_)
            return True

    def _assign(self, r, c, v):
        """填值并过滤行/列可行集, 记录 undo 轨迹. 矛盾返回 False."""
        # 对无约束行/列, 校验值未被占用(拉丁方); 有约束侧由可行排列集保证
        if (self.row_used[r] >> v) & 1 or (self.col_used[c] >> v) & 1:
            return False
        self.trail.append(("g", r, c, self.grid[r][c]))
        self.grid[r][c] = v
        self.trail.append(("u", 0, r, self.row_used[r]))
        self.row_used[r] |= 1 << v
        self.trail.append(("u", 1, c, self.col_used[c]))
        self.col_used[c] |= 1 << v
        if self.row_cur[r] is not None:
            self.trail.append(("lc", 0, r, self.row_cur[r], self.row_possible[r]))
            if not self._apply_filter("row", r, c, v):
                return False
        if self.col_cur[c] is not None:
            self.trail.append(("lc", 1, c, self.col_cur[c], self.col_possible[c]))
            if not self._apply_filter("col", c, r, v):
                return False
        # 松散行/列填满时校验可见性线索(紧致行/列由可行排列集保证)
        n = self.n
        if self.row_cur[r] is None and 0 not in self.grid[r]:
            row = self.grid[r]
            if self.left[r] and visible_count(row) != self.left[r]:
                return False
            if self.right[r] and visible_count(row[::-1]) != self.right[r]:
                return False
        if self.col_cur[c] is None:
            col = [self.grid[i][c] for i in range(n)]
            if 0 not in col:
                if self.top[c] and visible_count(col) != self.top[c]:
                    return False
                if self.bottom[c] and visible_count(col[::-1]) != self.bottom[c]:
                    return False
        return True

    def _undo_to(self, checkpoint):
        while len(self.trail) > checkpoint:
            item = self.trail.pop()
            if item[0] == "g":
                _, r, c, old = item
                self.grid[r][c] = old
            elif item[0] == "u":
                _, kind, idx_, old = item
                if kind == 0:
                    self.row_used[idx_] = old
                else:
                    self.col_used[idx_] = old
            else:  # lc
                _, kind, idx_, cur, poss = item
                if kind == 0:
                    self.row_cur[idx_] = cur
                    self.row_possible[idx_] = poss
                else:
                    self.col_cur[idx_] = cur
                    self.col_possible[idx_] = poss

    # ---------------- 候选域 ----------------
    def _cell_domain(self, r, c):
        """格子候选值(整数位掩码). 无约束侧用行列占用; 有约束侧用弧一致位掩码."""
        rp = self.row_possible[r]
        row_mask = (rp[c] if rp is not None else (~self.row_used[r]) & ~1)
        cp = self.col_possible[c]
        col_mask = (cp[r] if cp is not None else (~self.col_used[c]) & ~1)
        return row_mask & col_mask

    @staticmethod
    def _mask_count(mask):
        return bin(mask).count("1")

    @staticmethod
    def _mask_values(mask, n):
        return [v for v in range(1, n + 1) if (mask >> v) & 1]

    def _filter_line_by_cross(self, kind, i):
        """联合弧一致: 用相对方向各线当前可能值, 过滤本行/列可行排列.

        仅对紧致线(perm set 非 None)生效. 过滤使某格成为唯一值由传播第2步发现.
        记录 undo 轨迹.
        """
        n = self.n
        P = self.P
        if kind == "row":
            cur = self.row_cur[i]
            if cur is None:
                return True
            keep = cur
            for c in range(n):
                if self.grid[i][c]:
                    continue
                cp = self.col_possible[c]
                if cp is None:
                    continue
                mask = cp[i]
                sub = P[keep, c].astype(np.int32)
                keep = keep[((mask >> sub) & 1).astype(bool)]
            if len(keep) == 0:
                return False
            if len(keep) < len(cur):
                self.trail.append(("lc", 0, i, self.row_cur[i], self.row_possible[i]))
                self.row_cur[i] = keep
                self._recompute_possible("row", i)
            return True
        else:
            cur = self.col_cur[i]
            if cur is None:
                return True
            keep = cur
            for r in range(n):
                if self.grid[r][i]:
                    continue
                rp = self.row_possible[r]
                if rp is None:
                    continue
                mask = rp[i]
                sub = P[keep, r].astype(np.int32)
                keep = keep[((mask >> sub) & 1).astype(bool)]
            if len(keep) == 0:
                return False
            if len(keep) < len(cur):
                self.trail.append(("lc", 1, i, self.col_cur[i], self.col_possible[i]))
                self.col_cur[i] = keep
                self._recompute_possible("col", i)
            return True

    # ---------------- 传播 ----------------
    def _propagate(self):
        """强制填出所有唯一候选, 直至不动点. 矛盾返回 False."""
        n = self.n
        changed = True
        _it = 0
        while changed:
            _it += 1
            if _it > 300:
                raise TimeoutError
            changed = False
            # 1) 格子唯一候选
            for r in range(n):
                for c in range(n):
                    if self.grid[r][c]:
                        continue
                    dom = self._cell_domain(r, c)
                    if dom == 0:
                        return False
                    if dom & (dom - 1) == 0:  # 只有一个位
                        v = dom.bit_length() - 1
                        if not self._assign(r, c, v):
                            return False
                        changed = True
            # 2) 行/列在某个位置只剩一个可能值 (先做行↔列联合弧一致)
            for i in range(n):
                if self.row_cur[i] is not None:
                    if not self._filter_line_by_cross("row", i):
                        return False
                    for c in range(n):
                        if self.grid[i][c]:
                            continue
                        vals = self.row_possible[i][c]
                        if vals == 0:
                            return False
                        if vals & (vals - 1) == 0:
                            v = vals.bit_length() - 1
                            if not self._assign(i, c, v):
                                return False
                            changed = True
            for j in range(n):
                if self.col_cur[j] is not None:
                    if not self._filter_line_by_cross("col", j):
                        return False
                    for r in range(n):
                        if self.grid[r][j]:
                            continue
                        vals = self.col_possible[j][r]
                        if vals == 0:
                            return False
                        if vals & (vals - 1) == 0:
                            v = vals.bit_length() - 1
                            if not self._assign(r, j, v):
                                return False
                            changed = True
        return True

    # ---------------- 搜索 ----------------
    def _search(self):
        if time.time() - self._t0 > _TIME_BUDGET:
            raise TimeoutError
        self.nodes += 1
        n = self.n
        # MRV 选格
        best = None
        best_len = n + 1
        for r in range(n):
            for c in range(n):
                if self.grid[r][c]:
                    continue
                dom = self._cell_domain(r, c)
                if dom == 0:
                    return None
                cnt = self._mask_count(dom)
                if cnt < best_len:
                    best_len = cnt
                    best = (r, c)
                    if cnt == 1:
                        break
            if best_len == 1:
                break
        if best is None:
            return [list(row) for row in self.grid]

        r, c = best
        dom = self._cell_domain(r, c)
        vals = self._mask_values(dom, n)
        for v in vals:
            checkpoint = len(self.trail)
            if self._assign(r, c, v):
                if not self._propagate():
                    self._undo_to(checkpoint)
                    continue
                res = self._search()
                if res is not None:
                    return res
            self._undo_to(checkpoint)
        return None


def solve(n, top, bottom, left, right, given):
    """求解高楼谜题, 返回首个找到的 n×n 解矩阵; 无解/矛盾/超时返回 None."""
    if not (2 <= n <= 12):
        return None
    try:
        s = _Solver(n, list(top), list(bottom), list(left), list(right), given)
        if not s._propagate():
            return None
        return s._search()
    except Inconsistent:
        return None
    except TimeoutError:
        return None
    except RecursionError:
        return None
