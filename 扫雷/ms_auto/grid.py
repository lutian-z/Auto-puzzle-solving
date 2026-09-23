# -*- coding: utf-8 -*-
"""ms_auto.grid — 网格检测与图像对齐: 隔线 / 规整线 / 逐屏分析 / 滚动偏移 / 顶边锚定
"""
import cv2
import numpy as np

from .desktop import grab_screen, get_screen_size

DARK_GRAY = 130          # 隔线(深灰)阈值
SEP_THRESH = 0.50        # 某行/列被视为隔线所需暗像素占比
MIN_LINES = 3            # 至少检测到几条线才算有效

# 反锯齿柔化线(见 _resplit_bands): 浏览器小数格距会把一半网格线渲染成
# 跨两行的中灰双线(实测像素值约 134/146-159), 低于 DARK_GRAY 的实线阈值
# 就会整条漏检。格体最暗约 204(未翻开灰格), 因此 180 能盖住柔化线而
# 不会把格体算成线。
SOFT_DARK = 180          # 柔化网格线像素上界
SOFT_LINE_FRAC = 0.75    # 中线候选行所需柔暗像素占比
SOFT_SPLIT_MIN_PITCH = 16  # 物理可行性下限(非尺寸分档): 细分后子格须≥16px 才可分类

# 隔线检测

def _cluster_bands(indices, gap=4):
    """把连续索引聚成若干带, 返回每个带的中心位置(整数).

    gap 为允许的带内空隙; 一个"带"代表一条隔线的所有暗像素行/列,
    其中心即网格线所在位置.
    """
    if len(indices) == 0:
        return []
    bands = []
    start = prev = indices[0]
    for v in indices[1:]:
        if v - prev <= gap:
            prev = v
        else:
            bands.append(int(round((start + prev) / 2.0)))
            start = prev = v
    bands.append(int(round((start + prev) / 2.0)))
    return bands


def detect_h_bands(gray, thresh=SEP_THRESH):
    """检测横向隔线(整行深色占比高), 返回行位置列表."""
    h, w = gray.shape
    dark = (gray < DARK_GRAY).astype(np.uint8)
    frac = dark.sum(axis=1) / float(w)
    return _cluster_bands(np.where(frac > thresh)[0])


def detect_v_bands(gray, row_range=None, thresh=SEP_THRESH):
    """检测纵向隔线.

    row_range 指定统计用的行区间(棋盘纵向范围), 排除棋盘上下的背景,
    使纵向隔线检测更稳健.
    """
    h, w = gray.shape
    dark = (gray < DARK_GRAY).astype(np.uint8)
    if row_range is not None:
        y0, y1 = row_range
        y0, y1 = max(0, y0), min(h, y1)
        if y1 - y0 < 4:
            return []
        frac = dark[y0:y1, :].sum(axis=0) / float(y1 - y0)
    else:
        frac = dark.sum(axis=0) / float(h)
    return _cluster_bands(np.where(frac > thresh)[0])


def compute_pitch(positions):
    """隔线间距的中位数."""
    if positions is None or len(positions) < 2:
        return None
    diffs = np.diff(positions)
    return float(np.median(diffs))


def _filter_band_spacing(positions):
    """剔除间距异常的隔线带.

    棋盘的隔线间距应基本一致. 若检测到两条间距远小于主体间距的带(如棋盘
    边缘残带、界面元素), 保留主体间距更大的那条, 剔除异常者. 反复过滤直到
    间距一致.
    """
    if positions is None or len(positions) < 4:
        return list(positions)
    positions = list(positions)
    for _ in range(8):
        if len(positions) < 4:
            break
        diffs = np.array([positions[i + 1] - positions[i] for i in range(len(positions) - 1)])
        med = float(np.median(diffs))
        # 找到最小间距
        min_i = int(np.argmin(diffs))
        min_d = diffs[min_i]
        # 若最小间距远小于中位数(比如 < 0.6 中位数), 存在异常带
        if min_d < 0.6 * med:
            # 异常带相邻的两个间距: diffs[min_i] 与 diffs[min_i+1] (若存在)
            # 判断删除 positions[min_i] 还是 positions[min_i+1]
            # 看删除哪个能让相邻间距更接近 med
            left_d = diffs[min_i - 1] if min_i > 0 else None
            right_d = diffs[min_i + 1] if min_i + 1 < len(diffs) else None
            if left_d is None:
                # 左端出现“图像边缘 + 第一条真实网格线”时，应删除更靠外
                # 的图像边缘，而不是删掉第一条真实线。
                del positions[min_i]
            elif right_d is None:
                # 右端同理：保留倒数第二条真实网格线，删除贴近它的图像边缘。
                del positions[min_i + 1]
            else:
                # 比较: 删左 -> 新间距 = min_d + left_d; 删右 -> 新间距 = min_d + right_d
                # 选使新间距更接近 med 的
                if abs((min_d + left_d) - med) <= abs((min_d + right_d) - med):
                    del positions[min_i]
                else:
                    del positions[min_i + 1]
            continue
        break
    return positions


def _has_board_columns(gray, y0, y1):
    """区间 [y0, y1) 内是否仍有棋盘纵向网格结构.

    用于区分两类大间隔:
      - 漏检了若干行的真实棋盘线: 间隔区域仍是棋盘, 有规则、密集的纵向列线,
        且亮度处于棋盘格范围;
      - 棋盘下方的孤立深色线(浏览器窗口下沿/任务栏/桌面壁纸): 通常极暗或
        极亮, 且没有棋盘的列线密度。
    """
    if gray is None or y1 - y0 < 4:
        return False
    sub = gray[y0:y1, :]
    mean = float(sub.mean())
    # 棋盘格区域不是极暗或极亮的纯色块.
    if mean < 80 or mean > 235:
        return False
    width = gray.shape[1]
    vx = detect_v_bands(gray, row_range=(y0, y1))
    if len(vx) < MIN_LINES:
        return False
    gaps = np.diff(np.asarray(vx, dtype=float))
    if len(gaps) < 2:
        return False
    med = float(np.median(gaps))
    if med < 3:
        return False
    # 棋盘列线密度: (线数-1)*间距 ≈ 区域宽度(约 1 线/格). 桌面/壁纸上的
    # 零星竖线不满足该密度.
    density = (len(vx) - 1) * med / float(width)
    return density >= 0.35


def _trim_edge_outliers(positions, pitch, gray=None, max_ratio=1.5):
    """裁掉与棋盘网格不连续的首/尾孤立线带.

    浏览器窗口下沿、任务栏、桌面壁纸等会在棋盘下方(或上方)额外检测出一条
    深色横线。它与棋盘最后一条线的间距明显大于一格(> max_ratio*pitch),
    会被 count_grid_intervals 误算成多行, 也会被误当成“真实底边”。这里从
    两端裁掉这种孤立带。

    但若大间隔其实是“漏检了多行”造成的(间隔区域仍是棋盘, 有纵向列线),
    则保留尾部带, 由 count_grid_intervals 把大间隔还原成多行。

    安全护栏: 至少保留 MIN_LINES 条带, 避免把整个棋盘裁没。
    """
    if positions is None or len(positions) < 3 or pitch <= 0:
        return list(positions) if positions else []
    bands = sorted(set(int(round(v)) for v in positions))
    limit = max_ratio * float(pitch)
    while len(bands) > MIN_LINES:
        tail_gap = bands[-1] - bands[-2]
        head_gap = bands[1] - bands[0]
        if tail_gap > limit and not _has_board_columns(gray, bands[-2], bands[-1]):
            bands.pop()
            continue
        if head_gap > limit and not _has_board_columns(gray, bands[0], bands[1]):
            bands.pop(0)
            continue
        break
    return bands


def count_grid_intervals(positions, pitch):
    """根据相邻检测线之间跨过的格数统计总格数。

    浏览器缩放会把理论上的非整数格宽分配成 40/41/42 等不同像素宽度，
    因而不能用 ``总跨度 / 中位数间距`` 再四舍五入；40 格可能因此变成
    39 格。逐段计数既保留这些正常抖动，也能在漏掉一条线时把约 2*pitch
    的大间隔识别成两个格子。
    """
    if positions is None or len(positions) < 2 or pitch <= 0:
        return 0
    total = 0
    for left, right in zip(positions, positions[1:]):
        gap = float(right - left)
        total += max(1, int(round(gap / float(pitch))))
    return total


def regularize_lines(positions, pitch, n, lo=0, hi=None):
    """生成 n+1 条规整网格线, 尽量贴近检测到的隔线.

    检测到的真实线优先原样保留。只有相邻两条检测线的间隔跨过多个格子
    时，才在这两个真实端点之间做局部插值补线。不能从某个起点按固定
    ``pitch`` 一路外推，否则浏览器缩放造成的 40/42px 宽度分配会逐列
    累积相位误差，最终把单元格切到相邻列。
    """
    if positions is None or len(positions) == 0 or pitch <= 0:
        return None

    positions = sorted(set(int(round(v)) for v in positions))
    out = [positions[0]]
    for left, right in zip(positions, positions[1:]):
        steps = max(1, int(round((right - left) / float(pitch))))
        for step in range(1, steps + 1):
            # 以两条真实检测线为端点局部均分；最后一点严格等于 right。
            value = left + (right - left) * step / float(steps)
            out.append(int(round(value)))

    expected = n + 1
    if len(out) < expected:
        # 极端情况下末端线不可见但调用方已有更可靠的 n，才允许从最后
        # 一段的实际平均格宽向外补齐；正常 analyze_screen 不会走到这里。
        actual_pitch = ((out[-1] - out[0]) / float(max(1, len(out) - 1))
                        if len(out) > 1 else pitch)
        while len(out) < expected:
            out.append(int(round(out[-1] + actual_pitch)))
    elif len(out) > expected:
        out = out[:expected]
    if hi is not None:
        out = [min(max(v, lo), hi) for v in out]
    return out


def _band_thickness(gray, center, axis="h", width=8):
    """测量某条隔线的厚度(暗带宽度), 用于识别顶部/底部边框."""
    h, w = gray.shape
    half = width
    if axis == "h":
        y0, y1 = max(0, center - half), min(h, center + half + 1)
        seg = gray[y0:y1, max(0, w // 4):w - w // 4]
        dark_rows = [(seg[i] < DARK_GRAY).mean() > 0.5 for i in range(seg.shape[0])]
    else:
        x0, x1 = max(0, center - half), min(w, center + half + 1)
        seg = gray[max(0, h // 4):h - h // 4, x0:x1]
        dark_rows = [(seg[:, i] < DARK_GRAY).mean() > 0.5 for i in range(seg.shape[1])]
    # 找含 center 的连续暗带长度
    best = 0
    for i in range(len(dark_rows)):
        if dark_rows[i]:
            j = i
            while j + 1 < len(dark_rows) and dark_rows[j + 1]:
                j += 1
            if i <= half <= j:      # 包含中心
                best = max(best, j - i + 1)
            i = j + 1
    return best


def _soft_midline_positions(gray, bands, lo, hi, axis):
    """尝试把每条带间半格距处的柔化网格线补出来.

    返回 (new_bands, ratio): ratio 为带间隔中检测到柔化中线的比例。
    axis='h': bands 为横线 y 列表, lo/hi 为统计用的 x 区间;
    axis='v': 同理交换。
    """
    if len(bands) < 2 or hi - lo < 8:
        return list(bands), 0.0
    lo, hi = max(0, lo), min(gray.shape[1 if axis == "h" else 0], hi)
    if hi - lo < 8:
        return list(bands), 0.0
    sub = gray[:, lo:hi] if axis == "h" else gray[lo:hi, :]
    # 逐行/列统计"比格体暗"的像素占比: 网格线(含反锯齿柔化线)整条横跨
    # 棋盘, 占比≈1; 数字笔画/浮雕阴影只占一小段, 占比远小于阈值。
    frac = (sub < SOFT_DARK).mean(axis=1) if axis == "h" \
        else (sub < SOFT_DARK).mean(axis=0)
    new_bands = [bands[0]]
    n_gaps = len(bands) - 1
    n_found = 0
    for a, b in zip(bands[:-1], bands[1:]):
        gap = b - a
        if gap < SOFT_SPLIT_MIN_PITCH * 2:
            new_bands.append(b)
            continue
        mid = (a + b) // 2
        w0, w1 = max(0, mid - 4), min(len(frac), mid + 5)
        win = frac[w0:w1].astype(float)
        hit = win >= SOFT_LINE_FRAC
        if not hit.any():
            new_bands.append(b)      # 中点没有横贯棋盘的暗线: 不是漏检
            continue
        idx = np.where(hit)[0]
        weights = win[idx]
        center = float(np.average(w0 + idx, weights=weights))
        new_bands.append(int(round(center)))
        new_bands.append(b)
        n_found += 1
    return new_bands, (n_found / float(n_gaps) if n_gaps else 0.0)


def _resplit_soft_lines(gray, hy_raw, vx_raw, pitch):
    """小数格距反锯齿会把一半网格线渲染成柔化灰线, DARK_GRAY 检不到,
    pitch 被双倍高估(20x20 读成 10x10). 若过半带间中点存在柔化线,
    则把中线补进带列表并减半格距. 返回 (hy_raw, vx_raw, pitch, split_ok)."""
    if pitch < SOFT_SPLIT_MIN_PITCH * 2:
        return hy_raw, vx_raw, pitch, False
    h_new, rh = _soft_midline_positions(
        gray, hy_raw, vx_raw[0] + 4, vx_raw[-1] - 4, "h")
    v_new, rv = _soft_midline_positions(
        gray, vx_raw, hy_raw[0] + 4, hy_raw[-1] - 4, "v")
    if rh < 0.5 and rv < 0.5:
        return hy_raw, vx_raw, pitch, False
    if rh >= 0.5:
        hy_raw = h_new
    if rv >= 0.5:
        vx_raw = v_new
    ph = compute_pitch(hy_raw) or pitch / 2
    pv = compute_pitch(vx_raw) or pitch / 2
    return hy_raw, vx_raw, float(np.median([ph, pv])), True


# 全柔化线兜底: 自相关周期 + 边框锚定合成整张网格
# 小格距(约 23px)非整数渲染时, 全部内部网格线都会被反锯齿摊到 134~230 的
# 灰度区间: 没有任何一条线过 DARK_GRAY, 甚至没有一条过 SOFT_DARK(180),
# 逐线检测(含 _resplit_soft_lines)只剩棋盘边框。此时唯一可靠的信号是
# "行/列中位数亮度相对格体基线的暗谷"及其严格周期性: 用自相关定周期,
# 以检出的两条边框为锚点合成等距线列, 再逐线做相对谷验证。全部验证通过
# 才替换检测带; 任何一步不满足就原样失败, 绝不猜网格。

RESCUE_DIP = 12.0        # 线位相对谷深度要求(灰阶)
RESCUE_MIN_PITCH = 8.0
RESCUE_MAX_PITCH = 220.0
RESCUE_AC_CONF = 0.35    # 自相关峰最低置信度


def _best_axis_pitch(med, a, b):
    """边框锚定的周期估计: 真实网格的格数 k 必为整数且线列两端恰在两条边框
    上, 于是直接按 k 从细到粗扫描等距线列(linspace(a,b,k+1)), 取第一个通过
    逐线暗谷验证 + 半格错位对照的 k。
    (旧版整数滞后自相关在小数格距失效: 21.48px 的网格在 lag 21/22 都相差
    0.5px, 长跨度去相关, 只剩 43 的倍频假峰。)"
    """
    span = b - a
    kmax = int(min(200, span // RESCUE_MIN_PITCH))
    for k in range(kmax, 2, -1):
        lines = _synthesize_axis(med, a, b, k)
        if lines is not None:
            return lines, span / float(k)
    return None, None


def _dip_verified(med, lines, span_lo, span_hi):
    """对合成线列逐线做相对谷验证, 返回通过条数.

    线位 ±2px 内的最暗行, 需比两侧 5~9px 参照区的基线暗 RESCUE_DIP 以上。
    参照取两侧较暗一边的均值: 未翻开灰格(204)旁的线仍能被翻开白格侧拉出
    对比, 不会因单侧基线偏高而漏验; 边框线外侧参照越界时退化为内侧。
    """
    n = 0
    hi_lim = len(med) - 1
    for y in lines:
        yi = int(round(y))
        if yi < 1 or yi > hi_lim - 1:
            continue
        w0, w1 = max(0, yi - 2), min(hi_lim + 1, yi + 3)
        line_min = float(np.min(med[w0:w1]))
        c_avgs = []
        for ca, cb in ((yi - 9, yi - 4), (yi + 5, yi + 10)):
            ca, cb = max(span_lo, ca), min(span_hi, cb)
            if cb - ca >= 2:
                c_avgs.append(float(np.mean(med[ca:cb])))
        if not c_avgs:
            continue
        body = max(c_avgs)
        if body - line_min >= RESCUE_DIP:
            n += 1
    return n


def _synthesize_axis(med, a, b, k):
    """以 a/b 两条检出边框线为锚, 等距切成 k 格并逐线验证.

    返回 list(浮点位置) 或 None。验证 = 每条内线在其 ±2px 内有相对暗谷
    (比两侧 5~9px 参照基线暗 RESCUE_DIP 以上), 边框锚点天然通过; 另做
    半格错位对照(真网格下错半格应几乎无暗谷, 防止把倍频当周期)。
    """
    span = b - a
    if k < 3 or k > 200:
        return None
    p = span / float(k)
    if p < RESCUE_MIN_PITCH or p > RESCUE_MAX_PITCH:
        return None
    lines = [a + span * i / float(k) for i in range(k + 1)]
    ok = _dip_verified(med, lines[1:-1], a, b) + 2
    if ok < 0.93 * len(lines):
        return None
    off = [a + span * (i + 0.5) / float(k) for i in range(k)]
    ok_off = _dip_verified(med, off, a, b)
    if ok_off > 0.4 * k:
        return None
    return lines


def _bands_look_sparse(gray, hy_raw, vx_raw, pitch):
    """细(再)分之后, 检测带间距仍远大于边框锚定估计周期 => 大面积漏线."""
    h, w = gray.shape
    if not hy_raw or not vx_raw:
        return True
    med_h = np.median(gray[:, max(0, vx_raw[0] + 2):min(w, vx_raw[-1] - 2)], axis=1)
    med_v = np.median(gray[max(0, hy_raw[0] + 2):min(h, hy_raw[-1] - 2), :], axis=0)
    for med, bands in ((med_h, hy_raw), (med_v, vx_raw)):
        if len(bands) < 2:
            return True
        _, p = _best_axis_pitch(med, bands[0], bands[-1])
        if p is None:
            continue
        gp = compute_pitch(bands)
        if gp is not None and gp > 1.55 * p:
            return True
    return False


def _snap_lines_to_valleys(med, lines, radius=3):
    """把合成线吸附到附近最暗的真实线位(±radius 内取谷).

    等距合成相对浏览器的小数布局存在逐线累积的 ±1px 相位差; 小格距时
    未翻开格的浮雕高光只有 1px, 这 1px 相位差就能把高光挤出裁切窗,
    使整片未翻开格被误读成空白。吸附后谷心即线心, 裁切窗回到设计语义。
    """
    out = []
    hi = len(med) - 1
    for y in lines:
        yi = int(round(y))
        w0, w1 = max(0, yi - radius), min(hi + 1, yi + radius + 1)
        out.append(w0 + int(np.argmin(med[w0:w1])))
    return out


def _synthesize_grid_rescue(gray, hy_raw, vx_raw):
    """检测带太少/太疏时的兜底. 返回 (hy_new, vx_new, pitch) 或 None."""
    if len(hy_raw) < 2 or len(vx_raw) < 2:
        return None
    h, w = gray.shape
    med_h = np.median(gray[:, max(0, vx_raw[0] + 2):min(w, vx_raw[-1] - 2)], axis=1)
    med_v = np.median(gray[max(0, hy_raw[0] + 2):min(h, hy_raw[-1] - 2), :], axis=0)
    lines_h, ph = _best_axis_pitch(med_h, hy_raw[0], hy_raw[-1])
    lines_v, pv = _best_axis_pitch(med_v, vx_raw[0], vx_raw[-1])
    if lines_h is None or lines_v is None:
        return None
    # 已检出的其余带必须落进合成网格(排除把桌面杂线当边框锚点的情况)
    for bands, lines in ((hy_raw, lines_h), (vx_raw, lines_v)):
        for bnd in bands:
            if min(abs(bnd - lp) for lp in lines) > 2.5:
                return None
    # 方形棋盘: 两个方向的格数与跨度必须一致
    if abs(len(lines_h) - len(lines_v)) > 1:
        return None
    if abs((hy_raw[-1] - hy_raw[0]) - (vx_raw[-1] - vx_raw[0])) > \
            0.1 * max(hy_raw[-1] - hy_raw[0], 1.0):
        return None
    # 吸附到真实暗线位(见 _snap_lines_to_valleys)
    hy_new = _snap_lines_to_valleys(med_h, lines_h)
    vx_new = _snap_lines_to_valleys(med_v, lines_v)
    if len(set(hy_new)) < len(lines_h) or len(set(vx_new)) < len(lines_v):
        return None            # 吸附导致重线: 相位不可信, 放弃兜底
    return [int(v) for v in hy_new], [int(v) for v in vx_new], float(np.mean([ph, pv]))


def analyze_screen(img_bgr):
    """分析一屏画面, 返回网格结构信息 dict; 失败返回 None.

    返回:
      n       列数(棋盘尺寸, 方形)
      m       本屏可见行数
      vx      规整后的纵向网格线 (n+1 个)
      hy      规整后的横向网格线 (m+1 个)
      pitch   格子间距(像素)
      hy_raw  检测到的横向隔线(用于滚动对齐)
      vx_raw  检测到的纵向隔线
      top_is_border  首条横线是否为顶部边框(更厚)
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return None

    # 1. 先找横向隔线(整行暗)
    hy_raw = detect_h_bands(gray)
    if len(hy_raw) < 2:
        return None

    # 2. 在棋盘纵向范围内统计纵向隔线
    row_lo, row_hi = hy_raw[0], hy_raw[-1]
    vx_raw = detect_v_bands(gray, row_range=(row_lo, row_hi))
    if len(vx_raw) < 2:
        vx_raw = detect_v_bands(gray)
    if len(vx_raw) < 2:
        return None

    # 1.5 全柔化线兜底: 边框之外一条内部线都检不到时, 尝试
    # "自相关周期 + 边框锚定"合成整张网格(见 _synthesize_grid_rescue)。
    synthesized = False
    pitch = None
    if len(hy_raw) < MIN_LINES or len(vx_raw) < MIN_LINES:
        syn = _synthesize_grid_rescue(gray, hy_raw, vx_raw)
        if syn is None:
            return None
        hy_raw, vx_raw, pitch = syn
        synthesized = True

    if not synthesized:
        # 3. 间距(先剔除异常近的列带, 避免边界干扰)
        # 列带的真实间距应基本一致(约等于 pitch); 异常近的带(如边缘残带)会拉低中位数
        vx_raw = _filter_band_spacing(vx_raw)
        hy_raw = _filter_band_spacing(hy_raw)
        if len(vx_raw) < MIN_LINES or len(hy_raw) < MIN_LINES:
            return None

        pitch_h = compute_pitch(hy_raw)
        pitch_v = compute_pitch(vx_raw)
        if pitch_h is None or pitch_v is None:
            return None
        pitch = float(np.median([pitch_h, pitch_v]))
        if pitch < 3:
            return None

        # 3.2 反锯齿柔化线细分: 小数格距时一半网格线被渲染成中灰双线,
        # DARK_GRAY 检不到会让 pitch 双倍高估(20x20 读成 10x10), 必须先补线。
        hy_raw, vx_raw, pitch, split_ok = _resplit_soft_lines(
            gray, hy_raw, vx_raw, pitch)

        # 3.5 裁掉与棋盘网格不连续的首/尾孤立横线(浏览器窗口下沿/任务栏/桌面
        # 壁纸上的深色线)。否则大间隔会被 count_grid_intervals 误算成多行, 也
        # 会被误当成“真实底边”, 导致“首屏可见 9 行”之类错误。
        hy_raw = _trim_edge_outliers(hy_raw, pitch, gray=gray)
        if len(hy_raw) < MIN_LINES:
            return None
        # 用裁剪后的棋盘纵向范围重新检测列线, 避免下方桌面/窗口线污染列线统计。
        row_lo, row_hi = hy_raw[0], hy_raw[-1]
        vx_raw = detect_v_bands(gray, row_range=(row_lo, row_hi))
        vx_raw = _filter_band_spacing(vx_raw)
        if len(vx_raw) < MIN_LINES:
            return None
        # 重检测得到的列线同样只含实线, 需按同一判据再补一次柔化线。
        v_new, rv = _soft_midline_positions(
            gray, vx_raw, int(hy_raw[0]) + 4, int(hy_raw[-1]) - 4, "v")
        if rv >= 0.5:
            vx_raw = v_new
        pitch_h = compute_pitch(hy_raw)
        pitch_v = compute_pitch(vx_raw)
        if pitch_h is None or pitch_v is None:
            return None
        pitch = float(np.median([pitch_h, pitch_v]))
        if pitch < 3:
            return None

        # 3.6 稀疏后检: 细分之后带间距仍远大于自相关周期 => 大面积漏线,
        # 与 1.5 同款合成兜底(此时边框通常检得较全, 锚点可靠)。
        if _bands_look_sparse(gray, hy_raw, vx_raw, pitch):
            syn = _synthesize_grid_rescue(gray, hy_raw, vx_raw)
            if syn is not None:
                hy_raw, vx_raw, pitch = syn
                synthesized = True

    # 4. 列数(方形棋盘, 行数目标=列数)。逐间隔统计，避免缩放产生的
    # 交替像素宽度使 40 格被“总跨度/中位数”误算为 39 格。
    n = count_grid_intervals(vx_raw, pitch)
    if n < 1 or n > 200:
        return None
    n_count = len(vx_raw) - 1
    if n_count != n:
        pass  # 允许漏检, 以跨度推算为准(打印警告由调用方处理)

    # 5. 可见完整行数，同样逐间隔统计。
    m = count_grid_intervals(hy_raw, pitch)
    if m < 1 or m > 200:
        return None

    # 6. 规整网格线
    vx = regularize_lines(vx_raw, pitch, n, 0, w - 1)
    hy = regularize_lines(hy_raw, pitch, m, 0, h - 1)
    if vx is None or hy is None:
        return None

    # 7. 棋盘外边框判断(比普通内部隔线更厚)。不仅记录顶部，还记录
    # 所有可能的底边及它在当前屏之前跨过的完整行数。网页视口底沿有时
    # 也会形成一条深色横线；要求候选线下方仍留有少量截图空间，避免把
    # 屏幕边缘误当成棋盘底边。
    line_thickness = [_band_thickness(gray, p, "h") for p in hy_raw]
    top_thick = line_thickness[0]
    med_thick = float(np.median(line_thickness[1:-1] or [top_thick]))
    # 边框比普通隔线明显更厚(宽松阈值, 素材中边框9-12px, 隔线6px)
    top_is_border = top_thick >= med_thick + 2
    border_threshold = max(3.0, med_thick + 1.5)
    min_bottom_margin = max(4.0, pitch * 0.15)
    bottom_border_rows = []
    bottom_border_details = []
    rejected_bottom_borders = []
    for i in range(1, len(hy_raw)):
        margin_below = (h - 1) - hy_raw[i]
        if line_thickness[i] >= border_threshold:
            gap = float(hy_raw[i] - hy_raw[i - 1])
            crossed = max(1, int(round(gap / pitch)))
            per_row_gap = gap / crossed
            full_ratio = per_row_gap / pitch
            local_rows = count_grid_intervals(hy_raw[:i + 1], pitch)
            detail = {
                "local_rows": local_rows,
                "y": int(hy_raw[i]),
                "thickness": int(line_thickness[i]),
                "margin_below": float(margin_below),
                "last_gap": gap,
                "full_ratio": float(full_ratio),
            }
            # 真底边前的一格必须具有接近完整 pitch 的高度。视口底部或
            # IDE/浏览器横线常会把只露出半格的区域封成“伪末行”；仅凭
            # 线条较粗和下方有余量不足以证明最后一行已经看全。
            if (margin_below >= min_bottom_margin and
                    full_ratio >= 0.84):
                bottom_border_rows.append(local_rows)
                bottom_border_details.append(detail)
            else:
                rejected_bottom_borders.append(detail)

    return {
        "n": n,
        "m": m,
        "vx": vx,
        "hy": hy,
        "pitch": pitch,
        "hy_raw": hy_raw,
        "vx_raw": vx_raw,
        "top_is_border": top_is_border,
        "top_thick": top_thick,
        "med_thick": med_thick,
        "line_thickness": line_thickness,
        "bottom_border_rows": bottom_border_rows,
        "bottom_border_details": bottom_border_details,
        "rejected_bottom_borders": rejected_bottom_borders,
        "bottom_is_border": m in bottom_border_rows,
    }


# 滚动偏移计算(图像对齐)

def match_scroll_delta(prev_hy, cur_hy, pitch, bin_width=2.0):
    """估算两屏之间的垂直滚动偏移像素.

    依据: 相邻两屏必然有一部分网格线重合, 相同网格线在两屏中的 y 坐标差
    即滚动偏移. 对全部"可能匹配"的线对统计差值直方图, 取峰值.

    返回 (delta, score):
      delta  滚动偏移像素(>0 表示画面下移/向上滚; 通常 >0)
      score  匹配到的线对数(越大越可信)
    """
    if prev_hy is None or cur_hy is None or len(prev_hy) < 1 or len(cur_hy) < 1:
        return 0.0, 0
    deltas = []
    for py in prev_hy:
        for cy in cur_hy:
            deltas.append(float(py - cy))
    if not deltas:
        return 0.0, 0
    lo, hi = min(deltas), max(deltas)
    if hi - lo < 1e-6:
        return deltas[0], len(deltas)

    bins = int(np.ceil((hi - lo) / bin_width)) + 1
    hist, edges = np.histogram(deltas, bins=bins, range=(lo, hi))
    best_bin = int(np.argmax(hist))
    bin_lo, bin_hi = edges[best_bin], edges[best_bin + 1]
    cluster = [d for d in deltas if bin_lo <= d < bin_hi]
    delta = float(np.mean(cluster)) if cluster else (bin_lo + bin_hi) / 2.0
    score = hist[best_bin]
    return delta, score




# 基于内容的图像对齐(滚动偏移)



def align_screens(prev_bgr, cur_bgr, grid_prev, grid_cur, pitch,
                  max_shift_ratio=1.5):
    """基于内容的垂直滚动偏移估计(多模板匹配法).

    思路: 滚动距离不一定是格子间距整数倍, 两屏网格线会错位. 从 prev 的多个
    垂直位置取"模板带", 在 cur 中做模板匹配, 取匹配分最高的那个, 其 y 位移
    即为滚动偏移.

    向下滚动时 prev 的下部会保留在画面内, 因此模板带从低到高覆盖全屏,
    且底部模板带优先参与.

    返回 (delta, score):
      delta>0 表示 cur 内容相对 prev 上移 delta(即向下滚动).
    """
    if prev_bgr is None or cur_bgr is None:
        return 0.0, -1
    gp = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    hp, wp = gp.shape
    hc, wc = gc.shape
    if hp < 40 or hc < 40:
        return 0.0, -1

    # 棋盘区域(水平范围)
    def board_xrange(grid, w):
        if grid and (grid.get("vx") or grid.get("vx_raw")):
            vx = grid.get("vx") or grid["vx_raw"]
            x0, x1 = int(vx[0]), int(vx[-1])
            x0, x1 = max(0, x0), min(w, x1)
            if x1 - x0 > 8:
                return x0, x1
        return 0, w

    px0, px1 = board_xrange(grid_prev, wp)
    cx0, cx1 = board_xrange(grid_cur, wc)

    # 模板宽度: 棋盘中间 50%
    tw = int((px1 - px0) * 0.5)
    if tw < 8:
        tw = px1 - px0
    tx0 = px0 + (px1 - px0 - tw) // 2

    # 棋盘在 prev 图像内的纵向范围(用检测到的网格线 hy 界定)。这是滚动对齐
    # 能否成功的关键: 当浏览器窗口很小时, bbox 从用户顶边一直延伸到屏幕底部,
    # 其中很大一部分是棋盘下方的桌面/任务栏。若按整个 bbox 高度按比例放模板带,
    # 大多数带会落在棋盘之外(桌面), 内容匹配必然失败。用棋盘 hy 界定后, 无论
    # 可见多少行(7 行还是 14 行), 模板带都落在棋盘内部。
    b_y0, b_y1 = 0, hp
    if grid_prev is not None:
        _hy = grid_prev.get("hy") or grid_prev.get("hy_raw")
        if _hy is not None and len(_hy) >= 2:
            b_y0 = max(0, int(_hy[0]))
            b_y1 = min(hp, int(_hy[-1]))
    if b_y1 - b_y0 < 30:
        b_y0, b_y1 = 0, hp
    bh = b_y1 - b_y0

    # 模板带高度: 棋盘高度的 20%(足够覆盖一两行内容, 又不会太平滑).
    th = int(bh * 0.20)
    if th < 8:
        th = max(8, hp // 3)
    if th > bh // 2:
        th = max(8, bh // 2)

    # 降采样加速
    scale = min(1.0, 700.0 / max(hp, hc))

    # 模板带位置: 均匀分布在棋盘纵向范围内(不依赖可见行数), 下移方向稍密.
    tpl_positions = []
    for i in range(4):
        frac = (i + 0.5) / 4.0
        y = int(b_y0 + bh * frac - th / 2)
        if 0 <= y <= hp - th:
            tpl_positions.append(y)
    if not tpl_positions:
        y = int(b_y0 + bh // 2 - th // 2)
        if 0 <= y <= hp - th:
            tpl_positions.append(y)
    if not tpl_positions:
        tpl_positions = [max(0, min(hp - th, hp // 3))]

    best = None
    for ty0 in tpl_positions:
        tpl = gp[ty0:ty0 + th, tx0:tx0 + tw]
        if tpl.size == 0:
            continue
        if scale < 1.0:
            tpl_s = cv2.resize(tpl, (max(1, int(tw * scale)), max(1, int(th * scale))),
                               interpolation=cv2.INTER_AREA)
            gc_s = cv2.resize(gc, (max(1, int(wc * scale)), max(1, int(hc * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            tpl_s = tpl
            gc_s = gc
        if tpl_s.shape[0] >= gc_s.shape[0] or tpl_s.shape[1] >= gc_s.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(gc_s, tpl_s, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        _, mv, _, mloc = cv2.minMaxLoc(res)
        best_y = mloc[1] / scale
        d = ty0 - best_y
        if best is None or mv > best[0]:
            best = (float(mv), d)

    if best is None:
        return 0.0, -1
    return best[1], best[0]






def _screens_identical(a, b, grid, max_mean_diff=3.0):
    """判断两屏棋盘区域是否几乎完全一致(页面没有滚动).

    只用于识别“页面已到滚动极限/滚动无效”这一确定状态: 棋盘区域平均像素差
    必须极小。棋盘内容以未翻开格为主时, 滚动一整行后相邻行内容不同, 平均
    像素差通常大于 10(实测), 因此不会被误判为“没动”; 而真正没动时该差
    接近 0。网格线相位在整屏滚动时会回到原位, 不能作为“没动”的依据。
    """
    if a is None or b is None:
        return False
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        return False
    vx = grid.get("vx") or grid.get("vx_raw")
    hy = grid.get("hy") or grid.get("hy_raw")
    if not vx or not hy:
        return False
    x0, x1 = int(vx[0]), int(vx[-1])
    y0, y1 = int(hy[0]), int(hy[-1])
    x0, x1 = max(0, x0), min(ga.shape[1], x1)
    y0, y1 = max(0, y0), min(ga.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    diff = np.abs(ga[y0:y1, x0:x1].astype(int) - gb[y0:y1, x0:x1].astype(int))
    return float(diff.mean()) < max_mean_diff


# 滚动偏移(内容互相关 + 网格线精修)

def compute_scroll_offset(prev_img, cur_img, prev_grid, cur_grid, prev_hy, cur_hy, pitch):
    """计算 cur 相对 prev 的滚动偏移(向下滚为正).

    以内容模板匹配为准; 若匹配分过低, 回退到网格线差值。

    旧实现回退时只改了 delta 却把内容匹配的低分原样返回, 导致调用方因
    “score < 0.6” 把网格线给出的可靠偏移误判为“无法对齐”。这里把网格线
    匹配到的线对数换算成与内容匹配一致的 0..1 置信度一并返回。
    """
    delta, score = align_screens(prev_img, cur_img, prev_grid, cur_grid, pitch)
    if score < 0.6:
        d2, s2 = match_scroll_delta(prev_hy, cur_hy, pitch)
        if s2 >= 3 and abs(d2) >= 1:
            # 线对数转 0..1: 3 条及以上共用网格线即视为高置信.
            line_conf = min(1.0, s2 / 3.0)
            if line_conf > score:
                delta, score = d2, line_conf
    return delta, score

def refine_horizontal_bbox(bbox, stop, config, max_iter=6):
    """水平自适应微调: 把截屏区域向左右扩展到棋盘完整边界.

    用户点击的点只是大致范围, 若点到棋盘内侧(切开格子), 首/末列会被裁掉,
    导致列数推算偏小. 此函数逐步向外扩展截屏区域, 直到列数不再增加
    (即棋盘完整包含).

    返回 (refined_bbox, grid, img).
    """
    left, top, right, bottom = bbox
    scr_w, scr_h = get_screen_size()

    img, grid = None, None
    prev_n = None
    for _ in range(max_iter):
        img, grid = _grab_analyze_region((left, top, right, bottom))
        if grid is None:
            break
        cur_n = grid["n"]
        if prev_n is not None and cur_n <= prev_n:
            # 列数不再增加, 说明棋盘已完整
            break
        prev_n = cur_n
        pitch = grid["pitch"]
        vx = grid.get("vx_raw") or grid["vx"]
        if not vx:
            break
        margin_l = vx[0]
        margin_r = (right - left) - vx[-1]
        expand = 0
        if margin_l > 2:
            expand = max(expand, int(pitch * 1.1))
        if margin_r > 2:
            expand = max(expand, int(pitch * 1.1))
        if expand <= 0:
            break
        new_left = max(0, left - expand)
        new_right = min(scr_w, right + expand)
        if new_left == left and new_right == right:
            break
        left, right = new_left, new_right
    return (left, top, right, bottom), grid, img


def _grab_analyze_region(bbox):
    img = grab_screen(bbox)
    if img is None:
        return None, None
    grid = analyze_screen(img)
    return img, grid
def _board_line_thicknesses(img_bgr, hy_raw):
    """测每条已检测横线的暗带厚度(与 analyze_screen 的边框判据同一来源)."""
    gray = cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2GRAY)
    return [_band_thickness(gray, p, "h") for p in hy_raw]


def _left_border_top_y(gray, vx_raw, pitch):
    """左边框竖带内"持续暗带"的起点 = 棋盘顶边所在行.

    页面横线也会穿过左边框的 x 位置, 但只是几个像素高的短促暗段; 只有
    棋盘左边框会在该处形成贯穿整个棋盘高度的长暗带。取"该行本身为暗,
    且其后一个格距窗口内暗行占比过半"的第一行, 即棋盘顶边——对细线
    (缩放后边框不比隔线粗)、断续/虚线边框均稳健; 页面横线因窗口占比低
    被自然跳过。
    """
    if not vx_raw:
        return None
    h, w = gray.shape
    x = int(vx_raw[0])
    strip = gray[:, max(0, x - 1):min(w, x + 2)]
    if strip.size == 0 or h < 10:
        return None
    dark = ((strip < DARK_GRAY).any(axis=1)).astype(np.float64)
    win = max(8, int(round(pitch)))
    if h <= win:
        return None
    csum = np.concatenate([[0.0], np.cumsum(dark)])
    for y in range(0, h - win):
        # 该行是暗带起点(本身为暗), 且其后一个格距内持续为暗(占比过半)
        if dark[y] and csum[y + win] - csum[y] >= 0.6 * win:
            return y
    return None


def _anchor_board_top(bbox, grid, img):
    """把识别起点锚定到棋盘真实的加粗顶边框.

    框选(或点选)的顶边高于棋盘顶边框时, 顶边框之上会混入页面杂线或被
    裁掉一半的未见行; 它们与边框的间隔会被 count_grid_intervals 算成
    多余的"第 0 行", 使棋盘整体落到全局行 1..n, 此后 confirm_real_bottom
    要求的 "g0 + local_rows == n" 永远无法满足, 程序会不停下滚直到无进展
    报错退出。这里在已截画面内定位加粗顶边框并裁掉其上方内容:

      - 最佳加粗候选就是首条横线 -> 原样返回(点选贴边框的正常情形, 零改动)
      - 顶边框在画面中部 -> 多候选时用"左边框长暗带起点=棋盘顶边"定位
        真边框(可识别恰好压在边框上方一格的同厚页面粗线), 纯内存裁剪后
        用 analyze_screen 复检首线确为顶边框
      - 画面内没有顶边框候选且顶边上方还有屏幕空间 -> 向上扩大截屏范围
        再找(页面静止, 不滚动; 框选完整题目时顶边框必在屏幕内)
      - 无法锚定时原样返回, 由下游"确认真实底边"安全门兜底
    """
    if grid is None or img is None or img.size == 0:
        return bbox, grid, img

    left, top, right, bottom = bbox
    pitch = float(grid.get("pitch") or 0.0)
    if pitch <= 0:
        return bbox, grid, img

    cur_bbox, cur_grid, cur_img = bbox, grid, img
    for _ in range(5):
        hy = [int(p) for p in (cur_grid.get("hy_raw") or [])]
        if len(hy) < MIN_LINES:
            break
        thickness = _board_line_thicknesses(cur_img, hy)
        interior = thickness[1:-1] or thickness
        border_thr = max(3.0, float(np.median(interior)) + 1.5)
        lo, hi = 0.8 * pitch, 1.25 * pitch
        n_guess = int(cur_grid.get("n") or 0)

        def run_below(idx):
            """该线下方连续约一格间距的行数; 真顶边框此值最长."""
            run = 0
            for a, b in zip(hy[idx:], hy[idx + 1:]):
                if lo <= (b - a) <= hi:
                    run += 1
                else:
                    break
            return run

        # 左边框长暗带起点 = 棋盘顶边(不依赖线条厚度, 细线/虚线均稳健)
        gray = cv2.cvtColor(np.ascontiguousarray(cur_img), cv2.COLOR_BGR2GRAY)
        top_y = _left_border_top_y(gray, cur_grid.get("vx_raw") or [], pitch)

        # 底边框下方没有行(run=0), 天然排除
        cands = [i for i, t in enumerate(thickness)
                 if t >= border_thr and run_below(i) >= 1]
        if cands:
            # 多个加粗候选时, 优先取位置与棋盘顶边吻合者, 再按运行长度
            if top_y is not None:
                ranked = sorted(
                    cands,
                    key=lambda i: (abs(hy[i] - top_y) <= 8, run_below(i), i),
                    reverse=True)
            else:
                ranked = sorted(cands, key=lambda i: (run_below(i), i),
                                reverse=True)
            if ranked[0] == 0 and (top_y is None or abs(hy[0] - top_y) <= 8):
                return cur_bbox, cur_grid, cur_img     # 首线即顶边框, 无需裁剪
            anchored = None
            for ci in ranked:
                by, bt = hy[ci], thickness[ci]
                crop = max(0, by - (int(bt) // 2 + 3))
                sub = np.ascontiguousarray(cur_img[crop:, :])
                g2 = analyze_screen(sub)
                # 复检: 首线确为加粗边框, 且其位置与棋盘顶边吻合
                # (防止裁到恰好压在边框上方一格的同厚页面粗线)
                if (g2 is not None and g2.get("top_is_border")
                        and g2.get("n") == cur_grid.get("n")
                        and (top_y is None
                             or abs(float(g2["hy_raw"][0]) - (top_y - crop))
                             <= 8)):
                    anchored = ((left, cur_bbox[1] + crop, right, bottom), g2, sub)
                    break
            if anchored is not None:
                return anchored
            return bbox, grid, img      # 候选复检失败: 放弃锚定, 保持原样
        # 无加粗线(部分缩放下边框不比隔线粗): 用棋盘顶边位置做几何锚定
        if top_y is not None and n_guess >= 2:
            geo = [i for i, yv in enumerate(hy)
                   if abs(yv - top_y) <= 8 and run_below(i) >= n_guess - 2]
            for gi in geo:
                if gi == 0:
                    return cur_bbox, cur_grid, cur_img   # 首线即顶边
                crop = max(0, hy[gi] - 6)
                sub = np.ascontiguousarray(cur_img[crop:, :])
                g2 = analyze_screen(sub)
                g2_hy = (g2 or {}).get("hy_raw") or []
                if (g2 is not None and g2.get("n") == cur_grid.get("n")
                        and len(g2_hy) >= 2
                        and g2_hy[0] <= max(12.0, pitch * 0.5)
                        and abs(g2_hy[0] - (top_y - crop)) <= 8
                        and 0.6 * pitch <= g2_hy[1] - g2_hy[0] <= 1.4 * pitch):
                    return ((left, cur_bbox[1] + crop, right, bottom), g2, sub)
            if geo:
                return bbox, grid, img      # 几何候选复检失败: 保持原样
        # 画面内没有顶边框: 上方有屏幕空间则扩大截屏范围再找(不滚动页面)
        cur_top = cur_bbox[1]
        if cur_top <= 0:
            break
        new_top = max(0, cur_top - int(max(pitch * 3.0, 120.0)))
        img2 = grab_screen((left, new_top, right, bottom))
        g2 = analyze_screen(img2) if img2 is not None else None
        if g2 is None:
            break
        cur_bbox, cur_grid, cur_img = (left, new_top, right, bottom), g2, img2
    return bbox, grid, img
