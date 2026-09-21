# -*- coding: utf-8 -*-
"""sat_engine.py — 马赛克(Mosaic) 精确求解引擎.

主引擎: SciPy HiGHS MILP(复用扫雷项目 _solve_global_milp 的"首解 + no-good
割再解"框架, 证明唯一性而非仅找可行解). 引入理由: 兄弟项目扫雷已用同框架
验证过同构约束("数字+周围计数"), scipy 应用广泛、API 稳定.
    变量: 全部 n² 格 x∈{0,1}(1=涂黑 — 注意数字格自身也是变量);
    约束: 每个数字格的 3×3 邻域(裁边)变量和 == 数字;
    唯一性: 解出 reference 后追加"排除该完整解"的割, 第二次无解即唯一;
    多解: 对两次解相同的格逐一固定取反探测, 只保留所有解一致的格.

回退: SciPy 缺失/异常/超时 → solver.solve(纯 Python 传播+回溯).
"""
import time

import numpy as np

import solver as _S

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
except ImportError:                               # 环境无 SciPy 时可回退
    milp = None


class InconsistentBoardError(_S.MosaicError):
    """题面约束互相矛盾(数字识别错误的一种明确信号)."""


def _build_matrix(n, nums):
    """用 numpy 一次性构造邻域约束稀疏矩阵.

    返回 (col_of, row_ids, col_ids, rhs, n_rows, cells) —
    col_of[r,c]: 格的变量列号; row_ids/col_ids/data: COO 三元组;
    rhs: 每条约束的数字; cells: 变量格列表(行优先).
    """
    cells = [(r, c) for r in range(n) for c in range(n)]
    col_of = np.full((n, n), -1, dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        col_of[r, c] = i

    offs = np.array(_S.NEIGH9, dtype=np.int64)
    dr, dc = np.nonzero(np.ones((n, n), dtype=bool))   # 每格都是约束中心候选
    # 只保留数字格
    is_num = np.zeros((n, n), dtype=bool)
    for (r, c) in nums:
        is_num[r, c] = True
    dr, dc = dr[is_num[dr, dc].reshape(-1)], dc[is_num[dr, dc].reshape(-1)]
    nbr_r = dr[:, None] + offs[None, :, 0]
    nbr_c = dc[:, None] + offs[None, :, 1]
    valid = (nbr_r >= 0) & (nbr_r < n) & (nbr_c >= 0) & (nbr_c < n)
    rr = np.clip(nbr_r, 0, n - 1)
    cc = np.clip(nbr_c, 0, n - 1)
    nb_col = col_of[rr, cc]
    nb_col = np.where(valid, nb_col, -1)

    row_ids = []
    col_ids = []
    rhs = []
    row_no = 0
    for k in range(len(dr)):
        r, c = int(dr[k]), int(dc[k])
        num = int(nums[(r, c)])
        cols = nb_col[k]
        cols = cols[cols >= 0]
        size = len(cols)
        if num < 0 or num > size:
            raise InconsistentBoardError(
                f"数字 ({r + 1},{c + 1})={num} 超出邻域大小 {size}, "
                f"题面非法")
        row_ids.extend([row_no] * size)
        col_ids.extend(cols.tolist())
        rhs.append(float(num))
        row_no += 1
    return cells, row_ids, col_ids, rhs, row_no


def solve_milp(n, nums, time_budget=20.0):
    """MILP 精确求解 + 唯一性判定. 返回 (grid, info).

    info["unique"]: True=已证明唯一 / False=多解(抛 MultiSolutionError 由
    上层处理更直观, 这里直接抛). 无解抛 NoSolutionError/InconsistentBoardError.
    """
    if milp is None:
        raise _S.SolverTimeoutError("no-scipy")
    t0 = time.monotonic()
    _S._validate_input(n, nums)
    cells, row_ids, col_ids, rhs, n_rows = _build_matrix(n, nums)
    m = len(cells)

    def remaining():
        return max(0.05, time_budget - (time.monotonic() - t0))

    constraints = []
    if n_rows:
        matrix = csr_matrix((np.ones(len(col_ids)), (row_ids, col_ids)),
                            shape=(n_rows, m), dtype=float)
        target = np.asarray(rhs, dtype=float)
        constraints.append(LinearConstraint(matrix, target, target))

    objective = np.zeros(m, dtype=float)
    integrality = np.ones(m, dtype=int)
    bounds = Bounds(np.zeros(m), np.ones(m))

    first = milp(objective, integrality=integrality, bounds=bounds,
                 constraints=constraints,
                 options={"time_limit": remaining(), "mip_rel_gap": 0.0})
    if not first.success:
        if first.status == 2:            # HiGHS: infeasible
            raise _S.NoSolutionError(
                "题目无解(全部数字约束合并后无可行解, 数字识别可能有误)")
        raise _S.SolverTimeoutError(
            f"MILP 首解超时(>{time_budget:.0f}s, {n}x{n})")
    reference = np.rint(first.x).astype(int)

    # no-good 割: sum(与 reference 不同的位) >= 1
    coeff = np.where(reference == 1, -1.0, 1.0)
    lower = 1.0 - float(reference.sum())
    no_good = LinearConstraint(
        csr_matrix(coeff.reshape(1, -1)),
        np.asarray([lower]), np.asarray([np.inf]))
    second = milp(objective, integrality=integrality, bounds=bounds,
                  constraints=constraints + [no_good],
                  options={"time_limit": remaining(), "mip_rel_gap": 0.0})

    if not second.success and second.status == 2:
        grid = [[bool(reference[r * n + c]) for c in range(n)]
                for r in range(n)]
        return grid, {"engine": "milp", "unique": True,
                      "solve_time": time.monotonic() - t0}
    if not second.success:
        raise _S.SolverTimeoutError(
            f"MILP 唯一性判定超时(>{time_budget:.0f}s, {n}x{n})")

    # 多解路径: 与扫雷一致 — 先排除两解差异格, 其余逐格固定取反探测
    alternate = np.rint(second.x).astype(int)
    ambiguous = set(np.flatnonzero(alternate != reference).tolist())
    pending = [i for i in range(m) if i not in ambiguous]
    for i in pending:
        if remaining() <= 0.06:
            break
        unit = csr_matrix(([1.0], ([0], [i])), shape=(1, m))
        opposite = 1.0 - float(reference[i])
        fixed = LinearConstraint(unit, np.asarray([opposite]),
                                 np.asarray([opposite]))
        probe = milp(objective, integrality=integrality, bounds=bounds,
                     constraints=constraints + [fixed],
                     options={"time_limit": remaining(),
                              "mip_rel_gap": 0.0})
        if probe.success:
            other = np.rint(probe.x).astype(int)
            ambiguous.update(
                np.flatnonzero(other != reference).tolist())
    if ambiguous:                        # 存在两个解都不同的格 → 真多解
        raise _S.MultiSolutionError(
            "题目存在多个合法解, 与唯一解谜题不符(识别可能有误)")
    grid = [[bool(reference[r * n + c]) for c in range(n)] for r in range(n)]
    return grid, {"engine": "milp", "unique": "uncertain-probe-timeout",
                  "solve_time": time.monotonic() - t0}


def solve_auto(n, nums, time_budget=60.0, log=None):
    """统一求解入口. 优先 MILP, 失败/缺失时回退纯 Python.

    返回 (status, grid, info), status ∈ "ok" / "unsat" / "timeout".
    多解视为 "unsat"(对唯一解谜题等价于题面自相矛盾, 上层提示重试).
    """
    budget = time_budget if time_budget is not None else 60.0
    t0 = time.time()
    if milp is not None:
        try:
            grid, info = solve_milp(n, nums, time_budget=budget * 0.7)
            info["elapsed"] = time.time() - t0
            return "ok", grid, info
        except _S.MultiSolutionError as e:
            if log:
                log.info("[求解] MILP: %s", e)
            return "unsat", None, {"engine": "milp",
                                   "elapsed": time.time() - t0,
                                   "reason": str(e)}
        except _S.NoSolutionError as e:
            if log:
                log.info("[求解] MILP: %s", e)
            return "unsat", None, {"engine": "milp",
                                   "elapsed": time.time() - t0,
                                   "reason": str(e)}
        except InconsistentBoardError as e:
            if log:
                log.info("[求解] MILP: %s", e)
            return "unsat", None, {"engine": "milp",
                                   "elapsed": time.time() - t0,
                                   "reason": str(e)}
        except Exception as e:                    # 模型异常 → 回退
            if log:
                log.info("[求解] MILP 异常(%s), 转纯 Python 回退", e)
    remain = budget - (time.time() - t0)
    if remain <= 0.5:
        return "timeout", None, {"engine": "none",
                                 "elapsed": time.time() - t0}
    try:
        grid, info = _S.solve(n, nums, time_budget=remain)
        info["elapsed"] = time.time() - t0
        return "ok", grid, info
    except _S.MultiSolutionError as e:
        return "unsat", None, {"engine": "python",
                               "elapsed": time.time() - t0,
                               "reason": str(e)}
    except _S.NoSolutionError as e:
        return "unsat", None, {"engine": "python",
                               "elapsed": time.time() - t0,
                               "reason": str(e)}
    except _S.SolverTimeoutError:
        return "timeout", None, {"engine": "python",
                                 "elapsed": time.time() - t0}
