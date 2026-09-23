# -*- coding: utf-8 -*-
"""recognizer.py — 帐篷与树题目图像识别(版面分析/网格提取/树识别/数字识别).

输入一张截图(用户粗框选区域, 无需像素级精确), 输出结构化题目 Puzzle:
    n / trees(树位置集合) / row_ct, col_ct(行列约束, None=无约束) /
    centers(每格中心像素坐标, 相对识别输入图) / board_bbox.

流程:
    1) 线掩码: HSV 中 "很暗(V<120) 或 中性较暗(S<60 且 V<215)" 的像素.
       内部网格线被抗锯齿成浅灰(V≈130~210), 树冠灰度≈141 与之重叠,
       必须用色度排除彩色元素(素材实测结论, 见 NOTES.md).
    2) 最大连通域 = 网格线网络, 外接框即题目真实边界(数字在框外,
       与网格间有空白, 不会并入).
    3) 预处理坐标回溯: 黑边预裁 → 去倾斜(Hough 偏差中位数) → 尺度
       归一化(格 <22px 或 >72px 时缩到 ~36px), 全部映射回输入图坐标系.
    4) 投影法找网格线: 墙掩码按行/列求和, 阈值 0.55×边长取峰段,
       峰段加权重心为线位置; 行/列线数必须相等(正方形), 间距需均匀.
       格内部 = 上线段尾+1 .. 下线段头(自适应线宽, 加粗外框不吃格).
    5) 树识别: 格内 HSV 绿冠像素占比(H35~85, S≥140, V≥120) > 12%
       判树; 真实分布两极分化(树≈0.3, 空格≈0), 灰区记告警.
    6) 数字识别: 四侧数字带按 棋盘外沿+1.7×格宽 取带, 带内按网格线切
       槽位逐格识别; 槽内单组件宽>高时按最小墨水列切分粘连字形, 单字形
       归一化后与 templates.npz 灰度软字形模板做 相关性/IoU 匹配. 低置信
       度触发二值化/形态学变体重认, 仍低则告警. 识别值做 行和=列和=树数
       一致性校验.

数字模板为预渲染的 0-9 灰度软字形数据文件 templates.npz(与代码同目录
分发, 任意 1~2 位数天然泛化), 非外部素材。
"""
import os

import cv2
import numpy as np

import solver

_TPL_FMT = 2          # 模板数据格式版本(灰度软字形=2)
_TPL_SIZE = 32        # 模板归一化尺寸(与 _normalize_glyph 一致)
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "templates.npz")
_TPL_CACHE = None     # 模块级缓存, 避免每次识别重复读盘


# 异常(任务书 第六步)

class RecognitionError(solver.TentsError):
    """识别失败(可重试), 带用户可读信息."""


class GridGeometryError(RecognitionError):
    """网格几何异常(非正方形/间距不均/规模异常等)."""


# 数字模板(预渲染数据文件, 与代码同目录分发)

def _load_templates():
    """从 templates.npz 加载 0-9 数字模板 {digit: [float32 掩码]}.

    数据文件与代码一同分发, 无外部素材依赖; 格式版本不符或缺失时报
    可读错误而非静默失败。
    """
    global _TPL_CACHE
    if _TPL_CACHE is not None:
        return _TPL_CACHE
    try:
        data = np.load(TEMPLATE_PATH, allow_pickle=False)
        fmt_ok = "_fmt" in data.files and int(data["_fmt"][0]) == _TPL_FMT
        tpls = {}
        if fmt_ok:
            for k in data.files:
                if k == "_fmt":
                    continue
                d = int(k.split("_")[0])
                arr = data[k].astype(np.float32)
                if arr.shape != (_TPL_SIZE, _TPL_SIZE):
                    fmt_ok = False
                    break
                tpls.setdefault(d, []).append(arr)
        if not (fmt_ok and all(d in tpls for d in range(10))):
            raise RecognitionError(
                "数字模板数据文件格式不正确(templates.npz), 请恢复随项目"
                "分发的模板文件")
    except RecognitionError:
        raise
    except Exception:
        raise RecognitionError(
            "无法读取数字模板文件 templates.npz, 请确认它与本代码位于"
            "同一目录")
    _TPL_CACHE = tpls
    return tpls


def _normalize_glyph(mask_u8, size):
    """字形(灰度或二值)等比缩放并居中到 size×size 画布, 返回 float32 0..1
    灰度图. 放大 INTER_CUBIC(边缘平滑), 缩小 INTER_AREA. 不做硬二值化 —
    保留抗锯齿灰度供互相关匹配. """
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return None
    g = mask_u8[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    sc = (size - 6) / max(h, w)
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
    g = cv2.resize(g, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = g
    return canvas.astype(np.float32) / 255.0


class DigitClassifier:
    """单字形 0-9 分类: 多字体模板 归一化互相关(灰度) + IoU(二值) 取最大,
    附次选分差. 灰度互相关对缩放/模糊的抗锯齿渐变更宽容, IoU 保底."""

    def __init__(self, templates):
        self.tpls = templates

    def classify(self, glyph_f32):
        """glyph_f32: 归一化 float32 掩码. 返回 (digit, score, margin)."""
        if glyph_f32 is None:
            return 0, 0.0, 0.0
        a = glyph_f32 > 0.5
        scores = {}
        for d, tpls in self.tpls.items():
            best = 0.0
            for t in tpls:
                try:
                    res = cv2.matchTemplate(glyph_f32, t,
                                            cv2.TM_CCOEFF_NORMED)
                    corr = float(res.max())
                except cv2.error:
                    corr = 0.0
                b = t > 0.5
                union = (a | b).sum()
                iou = float((a & b).sum()) / union if union else 0.0
                best = max(best, corr, iou)
            scores[d] = best
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        best_d, best_s = order[0]
        second = order[1][1] if len(order) > 1 else 0.0
        return best_d, best_s, best_s - second


# 基础掩码与版面分析

def _bg_level_v(hsv):
    """估计背景亮度: 中性像素(S<60) V 值的 90 分位."""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    vals = v[s < 60]
    if vals.size == 0:
        return 255.0
    return float(np.percentile(vals, 90))


def _line_mask(hsv):
    """网格线掩码: 中性较暗(S<60 且 V < 背景×0.87) 或 很暗(V<120).

    素材内部网格线被抗锯齿成浅灰(V≈130~210), 树冠灰度≈141 与之重叠,
    必须用色度排除彩色元素; 绿底格间的线抗锯齿混成暗绿(S≈99,V≈44),
    由 "很暗" 分支兜住. 阈值随背景亮度自适应(固定 V<215 在整体变暗的
    截图下会把背景吸进线掩码). 直方图为 黑/灰/白 三峰, Otsu 会按
    黑/非黑分裂而丢失浅灰线, 故不用 Otsu.
    注意: 此掩码不做中值去噪 —— 断续细线会被中值抹掉; 椒盐噪声的孤立
    点不会形成贯穿投影(0.55×边长阈值)所需的成片结构, 不影响 N 检测,
    数字带的去噪由 _band_ink 单独做.
    """
    thr = min(240.0, max(90.0, _bg_level_v(hsv) * 0.87))
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    return (((v < thr) & (s < 60)) | (v < 120)).astype(np.uint8)


def _largest_component(mask):
    """最大连通域(8邻). 返回 (组件掩码, 外接框 (x0,y0,x1,y1)) 或 None.

    闭运算前先剔除面积 <6 的孤立小组件: 椒盐噪声的点/短链会在闭运算中
    相互桥接并挂到网格组件上, 使外接框膨胀数十字素、吞进外侧数字带
    (6.png+椒盐1% 实测 bbox 左界 38→13). 网格线主体面积远大于阈值.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # 查表一次性清零小组件(旧实现逐组件做全图 labels==i 比较, 椒盐噪声
    # 下组件数可达数千 → O(组件数×W×H), 是实测最大热点; 输出与旧版逐
    # 比特一致, 与马赛克项目 9/1 优化同构)
    keep = np.ones(n, dtype=bool)
    if n > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6
    clean = np.where(keep[labels], mask, 0).astype(mask.dtype)
    closed = cv2.morphologyEx(
        clean, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n2 <= 1:
        return None, None
    i = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
    m = labels2 == i
    x = int(stats2[i, cv2.CC_STAT_LEFT])
    y = int(stats2[i, cv2.CC_STAT_TOP])
    w = int(stats2[i, cv2.CC_STAT_WIDTH])
    h = int(stats2[i, cv2.CC_STAT_HEIGHT])
    return m, (x, y, x + w, y + h)


def _find_board(hsv, cfg):
    """定位网格线网络组件与外接框(留 2px 边). 返回 (wall, bbox)."""
    h, w = hsv.shape[:2]
    lm = _line_mask(hsv)
    wall, (x0, y0, x1, y1) = _largest_component(lm)
    if wall is None:
        raise GridGeometryError("未能定位题目网格(框选区域内无足够线条)")
    bw, bh = x1 - x0, y1 - y0
    min_side = 3 * int(cfg["min_cell_px"])
    if bw < min_side or bh < min_side:
        raise GridGeometryError(
            f"定位到的网格过小({bw}x{bh}px), 请框选完整题目后重试")
    if not (0.5 <= bw / float(bh) <= 2.0):
        raise GridGeometryError(
            f"题目外接框宽高比异常({bw}x{bh}), 请确认框选的是正方形棋盘")
    m = 2
    x0c, y0c = max(0, x0 - m), max(0, y0 - m)
    x1c, y1c = min(w, x1 + m), min(h, y1 + m)
    return wall[y0c:y1c, x0c:x1c], (x0c, y0c, x1c, y1c)


def _skew_deg_fine(gray, cfg):
    """Hough 直线偏差中位数估计倾斜角(度, 绝对值). 复制自数墙项目.

    兼容不同 OpenCV 版本的 HoughLinesP 返回形状 (N,1,4) / (N,4).
    """
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120,
                            minLineLength=max(60, int(gray.shape[1] * 0.15)),
                            maxLineGap=6)
    if lines is None:
        return 0.0
    lines = np.asarray(lines).reshape(-1, 4)
    devs = []
    for x1, y1, x2, y2 in lines:
        a = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        if a <= 45:
            devs.append(a)
        elif a >= 135:
            devs.append(180.0 - a)
        else:
            devs.append(abs(a - 90.0))
    if not devs:
        return 0.0
    return float(np.median(devs))


def _rotate_gray(gray, ang_deg):
    """绕中心旋转(复制填充, 三次插值减少字形模糊)."""
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _rotate_hsv(hsv, ang_deg):
    h, w = hsv.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, 1.0)
    return cv2.warpAffine(hsv, m, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)


def _crop_dark_border(hsv, cfg):
    """四周为深色背景时裁到最大亮色区域. 返回 (hsv, (off_x, off_y)) ."""
    v = hsv[:, :, 2]
    h, w = v.shape
    thr = int(cfg["dark_v"])
    frame = np.concatenate([v[0, :], v[-1, :], v[:, 0], v[:, -1]])
    if (frame < thr).mean() < 0.6:
        return hsv, (0, 0)
    bright = (v >= thr).astype(np.uint8)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(bright, 4)
    if n <= 1:
        return hsv, (0, 0)
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh = int(stats[i, 0]), int(stats[i, 1]), \
        int(stats[i, 2]), int(stats[i, 3])
    if bw < w * 0.6 or bh < h * 0.6:
        return hsv, (0, 0)
    return hsv[y:y + bh, x:x + bw], (x, y)


# 网格线提取(投影法)

def _line_segs(proj, thr):
    """投影超阈值的连续峰段. 返回 [(start, end, 加权重心)]."""
    out = []
    i, n = 0, len(proj)
    while i < n:
        if proj[i] >= thr:
            j = i
            while j + 1 < n and proj[j + 1] >= thr:
                j += 1
            seg = proj[i:j + 1].astype(np.float64)
            idx = np.arange(i, j + 1)
            center = float((seg * idx).sum() / seg.sum())
            out.append((i, j, center))
            i = j + 1
        else:
            i += 1
    return out


def _fit_axis(centers, cell_est):
    """一维等差校验与最小二乘规整. 返回 (规整位置, step, 最大残差)."""
    pos = np.array(centers, dtype=np.float64)
    idx = np.arange(len(pos), dtype=np.float64)
    nn = len(idx)
    if nn < 2:
        return pos.tolist(), float(cell_est), 0.0
    sx, sy = idx.sum(), pos.sum()
    sxx = (idx * idx).sum()
    sxy = (idx * pos).sum()
    denom = nn * sxx - sx * sx
    step = ((nn * sxy - sx * sy) / denom) if abs(denom) > 1e-9 \
        else float(cell_est)
    intercept = (sy - step * sx) / nn
    fitted = intercept + step * idx
    residual = float(np.abs(pos - fitted).max())
    return fitted.tolist(), float(step), residual


def _grid_lines(wall, cfg):
    """从墙掩码投影提取网格线. 返回 (vsegs, hsegs, cell).

    vsegs/hlines: [(start, end, 加权重心)]; cell: 平均格宽.
    """
    h, w = wall.shape
    vp = wall.sum(axis=0)
    hp = wall.sum(axis=1)
    vsegs = _line_segs(vp, 0.55 * h)
    hsegs = _line_segs(hp, 0.55 * w)
    if len(vsegs) < 2 or len(hsegs) < 2:
        raise GridGeometryError(
            f"网格线检测不足(竖线{len(vsegs)}, 横线{len(hsegs)}), "
            f"请放大题目或重新框选")
    nv, nh = len(vsegs) - 1, len(hsegs) - 1
    if nv != nh:
        raise GridGeometryError(
            f"网格非正方形(列数{nv} != 行数{nh}), 请确认框选完整")
    # 间距均匀性: 相邻线中心距的变异系数
    vstep = np.diff([c for _s, _e, c in vsegs])
    hstep = np.diff([c for _s, _e, c in hsegs])
    cell = float(np.median(vstep))
    for name, steps in (("竖", vstep), ("横", hstep)):
        if len(steps) >= 3 and float(np.mean(steps)) > 0:
            cv_ = float(np.std(steps) / np.mean(steps))
            if cv_ > float(cfg["spacing_cv_max"]):
                raise GridGeometryError(
                    f"{name}线间距不均匀(变异系数 {cv_:.2f} > "
                    f"{cfg['spacing_cv_max']}), 题目可能被遮挡或混入其他内容")
    return vsegs, hsegs, cell


# 树识别

def _canopy_mask(hsv):
    """树冠掩码: 绿色高饱和 (H 35~85, S≥140, V≥120)."""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    return (h >= 35) & (h <= 85) & (s >= 140) & (v >= 120)


def _detect_trees(hsv_crop, hlines, vlines, cfg, warnings):
    """逐格检测: 树冠占比 → 树; 暗图标占比 → 残留帐篷(非题目内容).

    帐篷图标深色(V<110)占比实测 ≈0.11, 空格 ≈0(格线已被峰段边界避开),
    判定阈值 0.04. 树格优先(树冠/树干配色与帐篷不同, 且树是题目内容).
    返回 (trees, existing_tents).
    """
    canopy = _canopy_mask(hsv_crop)
    v_ch = hsv_crop[:, :, 2]
    trees = set()
    existing = set()
    n = len(hlines) - 1
    for r in range(n):
        ya = hlines[r][1] + 1
        yb = hlines[r + 1][0]
        for c in range(n):
            xa = vlines[c][1] + 1
            xb = vlines[c + 1][0]
            if yb <= ya or xb <= xa:
                continue
            frac = float(canopy[ya:yb, xa:xb].mean())
            if frac >= float(cfg["canopy_hi"]):
                trees.add((r, c))
            elif frac >= float(cfg["canopy_lo"]):
                trees.add((r, c))
                warnings.append(
                    f"格({r + 1},{c + 1}) 树冠占比 {frac:.2f} 处于灰区, "
                    f"已按树处理")
            else:
                dark = float((v_ch[ya:yb, xa:xb] < 110).mean())
                if dark > float(cfg["tent_dark_frac"]):
                    existing.add((r, c))
    if not trees:
        raise RecognitionError(
            "未在棋盘内识别到任何树, 请确认框选的是帐篷题目")
    return trees, existing


# 数字带识别

def _band_ink(gray, hsv, rect):
    """数字带的两种墨迹掩码(宽松/严格) + 中值去噪. 返回 (loose, tight, w, h).

    loose = 线掩码口径("中性较暗∪很暗"): 笔画完整, 但小字号数字的抗锯齿
            晕圈会被计入, 可能把 "6"/"0" 的字洞填成实心团;
    tight = 带内 Otsu(黑数字 vs 白底双峰): 保住字洞, 但极细笔画可能断裂。
    两者按槽位各读一次, 依分类置信度择优(_read_side)。
    数字笔画 1~4px, 中值3 去孤立噪点并平滑边缘。
    """
    x0, y0, x1, y1 = rect
    patch = hsv[y0:y1, x0:x1]
    v = patch[:, :, 2]
    s = patch[:, :, 1].astype(np.int32)
    loose = _line_mask(patch)
    vals = v[(s < 80) & (v > 0)].astype(np.uint8)
    thr, _ = cv2.threshold(vals, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU) \
        if vals.size > 16 else (120.0, None)
    if thr < 60 or thr > 245:
        thr = 120.0
    tight = ((v < thr) & (s < 80)).astype(np.uint8) * 255
    loose = cv2.medianBlur(loose * 255, 3)
    tight = cv2.medianBlur(tight, 3)
    return ((loose > 0).astype(np.uint8), (tight > 0).astype(np.uint8),
            x1 - x0, y1 - y0)


def _split_wide(mask):
    """宽组件按列墨水谷切分为多个字形."""
    ink = (mask > 0).sum(axis=0)
    w = len(ink)
    cuts = [0]
    i = 1
    while i < w - 1:
        if ink[i] == 0:
            j = i
            while j < w and ink[j] == 0:
                j += 1
            cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    cuts.append(w)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = mask[:, a:b]
        if (piece > 0).any():
            out.append(piece)
    return out if out else [mask]


def _split_connected(mask, min_area):
    """粘连字形切分: 无零谷时按最小墨水列切两段(两位数场景).

    小字号两位数(如 "13")间隙被抗锯齿填平后成为单组件, 宽高比
    (~1.2)达不到 _split_wide 的切分阈值; 在中部区域找墨水最少的
    列切开。切出的两片任一面积过小则放弃(不是两位数)。
    """
    ink = (mask > 0).sum(axis=0)
    w = len(ink)
    lo, hi = int(w * 0.28), int(w * 0.72)
    if hi - lo < 2:
        return None
    k = lo + int(np.argmin(ink[lo:hi]))
    a = mask[:, :k]
    b = mask[:, k:]
    if (a > 0).sum() < min_area or (b > 0).sum() < min_area:
        return None
    return [a, b]


def _read_digit_group(mask, clf, cfg, where, warnings):
    """识别一个数字组(1~2 个字形). 返回 (value 或 None, conf, low).

    字形按 x 坐标排序后拼接(连通域标签序不保证从左到右, "12"曾误拼"21").
    槽内单组件且宽>高时按最小墨水列尝试粘连切分(小字号两位数
    "13"/"11"的间隙常被抗锯齿填平, 详见 _split_connected).
    """
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(cfg["glyph_min_area"]):
            continue
        piece = (labels == i).astype(np.uint8) * 255
        xs = np.where(piece.any(axis=0))[0]
        comps.append((int(xs.min()), piece))
    if not comps:
        return None, 0.0, False
    comps.sort(key=lambda t: t[0])
    glyphs = []
    for _x0, piece in comps:
        h, w = piece.shape
        if w > 1.55 * max(h, 8):
            glyphs.extend(_split_wide(piece))
        else:
            glyphs.append(piece)
    if len(glyphs) == 1:
        g = glyphs[0]
        gh, gw = g.shape
        if gw > 1.0 * gh:
            two = _split_connected(g, int(cfg["glyph_min_area"]))
            if two:
                glyphs = two
    if len(glyphs) > 2:
        glyphs = glyphs[:2]
        warnings.append(f"{where} 字形数异常, 仅取前 2 位")
    digits = []
    confs = []
    low = False
    for gi, gmask in enumerate(glyphs):
        g = _normalize_glyph(gmask, _TPL_SIZE)
        d, score, _margin = clf.classify(g)
        if score < float(cfg["glyph_conf"]):
            low = True
        digits.append(d)
        confs.append(score)
    value = int("".join(str(d) for d in digits))
    return value, float(min(confs)) if confs else 0.0, low


def _read_digit_group_multi(mask, clf, cfg, where, warnings):
    """带变体重认的数字组识别(低置信度二次确认).

    变体: 原始 Otsu 二值 / 自适应阈值 / 轻度膨胀 / 轻度腐蚀 /
    3x放大后腐蚀1~2档(小字形放大插值后笔画偏粗, 腐蚀逼近模板细笔画).
    取各变体最高分结果; 最高分仍低于阈值则告警.
    """
    value, conf, low = _read_digit_group(mask, clf, cfg, where, warnings)
    if not low:
        return value, conf, False
    best = (value, conf)
    variants = []
    _, binary = cv2.threshold(mask, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)
    ad = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, 5)
    variants.append(ad)
    k = np.ones((2, 2), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    variants.append(cv2.dilate(mask, k))
    variants.append(cv2.erode(mask, k))
    try:
        big = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
        variants.append(cv2.erode(big, k3))
        variants.append(cv2.erode(big, k3, iterations=2))
    except cv2.error:
        pass
    for vi, vm in enumerate(variants):
        try:
            v2, c2, _low2 = _read_digit_group((vm > 0).astype(np.uint8) * 255,
                                              clf, cfg, where, warnings)
        except cv2.error:
            continue
        if v2 is not None and c2 > best[1]:
            best = (v2, c2)
    value, conf = best
    low = conf < float(cfg["glyph_conf"])
    if low:
        warnings.append(f"{where} 数字置信度低({conf:.2f}), 结果可能有误")
    return value, conf, low


def _read_side(gray, hsv, side, board, cell, lines, n, clf, cfg, warnings):
    """按槽位读取一侧数字带. 返回 {槽位索引: (value, conf, low)}.

    用规整网格线把数字带切成 n 个槽位(顶部/底部槽位的水平区间 =
    [竖线 c, 竖线 c+1], 左/右槽位的垂直区间同理), 每槽独立识别 ——
    从结构上消除"相邻数字被聚成一组"的歧义(两位数越宽越易与邻位
    合并, 自由聚组在实机缩放下会系统性错位)。
    side: 'top'/'bottom'/'left'/'right'; board: 墙外接框(处理图中坐标);
    lines: 该侧槽位轴向的规整网格线位置(与 board 同坐标系, 长 n+1).
    """
    x0, y0, x1, y1 = board
    H, W = gray.shape[:2]
    band = int(round(1.7 * cell))
    if side == "top":
        rect = (x0, max(0, y0 - band), x1, y0)
        cross = "x"
    elif side == "bottom":
        rect = (x0, y1, x1, min(H, y1 + band))
        cross = "x"
    elif side == "left":
        rect = (max(0, x0 - band), y0, x0, y1)
        cross = "y"
    else:
        rect = (x1, y0, min(W, x1 + band), y1)
        cross = "y"
    rx0, ry0, rx1, ry1 = rect
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return {}
    ink_l, ink_t, bw_, bh_ = _band_ink(gray, hsv, rect)
    axis_len = bw_ if cross == "x" else bh_
    out = {}
    # lines 与带矩形同一坐标系(原点均为棋盘框左上角), 不需再减原点;
    # 槽位轴向长度按带矩形截断(带可能被图像边缘裁短)
    for slot in range(n):
        lo = max(0, int(round(lines[slot])))
        hi = min(axis_len, int(round(lines[slot + 1])))
        if hi - lo < 3:
            continue
        sl = (slice(None), slice(lo, hi)) if cross == "x" \
            else (slice(lo, hi), slice(None))
        best = None
        for m in (ink_l[sl], ink_t[sl]):
            if int(m.sum()) < 12:
                continue
            value, conf, low = _read_digit_group_multi(
                m * 255, clf, cfg, f"{side}[{slot + 1}]", warnings)
            if value is None:
                continue
            if best is None or conf > best[1]:
                best = (value, conf, low)
        if best is not None:
            out[slot] = best
    return out


# 主入口

class Puzzle:
    """识别结果: 题目结构 + 几何."""

    def __init__(self, n, trees, row_ct, col_ct, centers, cell_w, cell_h,
                 board_origin, warnings, existing_tents=None,
                 scale=1.0, rot_ang=0.0):
        self.n = n
        self.trees = trees                  # set[(r,c)]
        self.row_ct = row_ct                # list[int|None]
        self.col_ct = col_ct
        self.centers = centers              # {(r,c): (x,y)} 相对识别输入图
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.board_origin = board_origin    # 棋盘外接框左上角(输入图坐标)
        self.warnings = warnings
        self.existing_tents = existing_tents or set()   # 棋盘残留帐篷
        self.scale = scale                  # 预处理参数(校准可比性判断用)
        self.rot_ang = rot_ang

    def describe(self):
        rt = "/".join("_" if v is None else str(v) for v in self.row_ct)
        ct = "/".join("_" if v is None else str(v) for v in self.col_ct)
        extra = f", 残留帐篷{len(self.existing_tents)}个" \
            if self.existing_tents else ""
        return (f"{self.n}x{self.n}, {len(self.trees)} 棵树, "
                f"行[{rt}] 列[{ct}]{extra}")


def recognize(img, cfg, log):
    """识别整道帐篷题目. img: BGR ndarray. 失败抛 RecognitionError 系."""
    if img is None or img.size == 0:
        raise RecognitionError("输入图像为空")
    warnings = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h0, w0 = hsv.shape[:2]

    # 1) 黑边预裁
    hsv, (off_x, off_y) = _crop_dark_border(hsv, cfg)
    if off_x or off_y:
        warnings.append("检测到四周深色边框, 已自动裁剪到纸面区域")
    h1, w1 = hsv.shape[:2]

    # 2) 去倾斜(迭代≤2轮: Hough 角度量化 0.25°, 大棋盘上残留会
    #      漂移投影; 多次重采样损失字形精度, 轮数须受限) ----
    rot_ang = 0.0
    gray1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
    for _it in range(2):
        ang0 = _skew_deg_fine(gray1, cfg)
        if ang0 < float(cfg["skew_warn_deg"]):
            break
        cand_g = _rotate_gray(gray1, ang0)
        cand_h = _rotate_hsv(hsv, ang0)
        ang1 = _skew_deg_fine(cand_g, cfg)
        if ang1 > ang0:                       # 方向反了: 反向转
            cand_g = _rotate_gray(gray1, -ang0)
            cand_h = _rotate_hsv(hsv, -ang0)
            rot_ang -= ang0
        else:
            rot_ang += ang0
        gray1, hsv = cand_g, cand_h
        warnings.append(f"题目存在 {abs(ang0):.1f}° 倾斜, 已自动转正")
    if abs(rot_ang) > float(cfg["skew_max_deg"]):
        raise GridGeometryError(
            f"题目倾斜约 {abs(rot_ang):.1f}° 超过上限 {cfg['skew_max_deg']}°, "
            f"请摆正窗口后重新框选")
    h2, w2 = gray1.shape[:2]

    # 3) 定位棋盘 + 尺度归一化
    wall, board = _find_board(hsv, cfg)
    cell0 = _estimate_cell(wall, board)
    scale = 1.0
    if 0 < cell0 and (cell0 < 22.0 or cell0 > 72.0):
        scale = max(0.4, min(2.5, 36.0 / cell0))
        gray1 = cv2.resize(gray1, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC if scale > 1
                           else cv2.INTER_AREA)
        hsv = cv2.resize(hsv, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST if scale < 1
                         else cv2.INTER_LINEAR)
        wall = cv2.resize(wall.astype(np.uint8), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        board = tuple(int(round(v * scale)) for v in board)
        warnings.append(
            f"单元格 {cell0:.0f}px 偏{'小' if scale > 1 else '大'}, "
            f"已按 {scale:.2f}x 归一化尺度")

    # 4) 网格线
    vsegs, hsegs, cell = _grid_lines(wall, cfg)
    n = len(vsegs) - 1
    if not (1 <= n <= int(cfg["max_board"])):
        raise GridGeometryError(
            f"网格规模异常(N={n}), 超出支持范围(1~{cfg['max_board']})")
    # 线位置最小二乘规整(消除像素抖动, 中心更稳)
    v_centers = [c for _s, _e, c in vsegs]
    h_centers = [c for _s, _e, c in hsegs]
    v_fit, step_x, res_x = _fit_axis(v_centers, cell)
    h_fit, step_y, res_y = _fit_axis(h_centers, cell)
    tol = 0.25 * max(step_x, step_y)
    if res_x > tol or res_y > tol:
        warnings.append(
            f"网格线位置残差偏大(x={res_x:.1f}, y={res_y:.1f}px), 已按等差规整")

    # 5) 树识别(顺带检测残留帐篷)
    bx0, by0, bx1, by1 = board
    hsv_crop = hsv[by0:by1, bx0:bx1]
    # 线段坐标转换到 board 裁剪系
    vsegs_c = [(s, e, c - bx0) for s, e, c in vsegs]
    hsegs_c = [(s, e, c - by0) for s, e, c in hsegs]
    trees, existing = _detect_trees(hsv_crop, hsegs_c, vsegs_c,
                                    cfg, warnings)
    if existing:
        warnings.append(
            f"棋盘上检测到 {len(existing)} 个残留帐篷(非题目内容), "
            f"作答前将自动清除")

    # 6) 数字识别(按槽位)
    clf = DigitClassifier(_load_templates())
    side_data = {}
    for side in ("top", "bottom", "left", "right"):
        lines = v_fit if side in ("top", "bottom") else h_fit
        found = _read_side(gray1, hsv, side, board, cell, lines, n, clf,
                           cfg, warnings)
        if found:
            side_data[side] = found

    def resolve(sides):
        """合并一侧或多侧的槽位识别值并解决左右冲突."""
        values = {}
        for side in sides:
            for slot, (value, _conf, _low) in side_data.get(side, {}).items():
                prev = values.get(slot)
                if prev is not None and prev != value:
                    raise RecognitionError(
                        f"第{slot + 1}格两侧数字冲突({prev} vs {value}), "
                        f"请重新框选")
                values[slot] = value
        return [values.get(i) for i in range(n)]

    row_ct = resolve(("left", "right"))
    col_ct = resolve(("top", "bottom"))
    if all(v is None for v in row_ct) and all(v is None for v in col_ct):
        raise RecognitionError(
            "未识别到任何行列约束数字, 请确认框选范围包含棋盘外侧数字")
    for name, arr in (("行", row_ct), ("列", col_ct)):
        for i, v in enumerate(arr):
            if v is not None and v > n:
                raise RecognitionError(
                    f"{name}约束 {name}{i + 1}={v} 超出网格规模 {n}, "
                    f"数字识别可能有误")

    # 7) 一致性校验
    n_trees = len(trees)
    if all(v is not None for v in row_ct) and \
            sum(row_ct) != n_trees:
        raise RecognitionError(
            f"行约束之和({sum(row_ct)}) != 树数({n_trees}), 数字识别可能有误")
    if all(v is not None for v in col_ct) and \
            sum(col_ct) != n_trees:
        raise RecognitionError(
            f"列约束之和({sum(col_ct)}) != 树数({n_trees}), 数字识别可能有误")

    # 坐标回溯映射到输入图
    # 格线/格中心的坐标是"墙裁剪坐标系"(裁剪原点=棋盘外接框左上角),
    # 输出的 centers/board_origin 必须加回裁剪原点映射到输入图坐标系!
    # (曾遗漏导致所有点击整体左上偏移约一格 — 实机作答全面错位的根因;
    # 数墙原实现是 rows_y[r]+origin[1], 重写投影法时丢失了这一步。)
    # 旋转分支方向(实测往返验证): warpAffine(gray, getRotationMatrix2D
    # (c, θ)) 的内容移动是 q = M(θ)·p, 因此"转正图→原图"要用 M(-θ)。
    # 旧注释引用文档公式 dst(x,y)=src(Mx) 推出"不能取逆"是符号方向理解
    # 反了, 会把偏差翻倍(3° 时约 38px); 该分支仅在 ≥0.7° 倾斜时执行。
    def to_input(x, y):
        if scale != 1.0:
            x, y = x / scale, y / scale
        if rot_ang:
            a = -np.deg2rad(rot_ang)
            ca, sa = np.cos(a), np.sin(a)
            cx, cy = w2 / 2.0, h2 / 2.0
            nx = ca * x + sa * y + (1 - ca) * cx - sa * cy
            ny = -sa * x + ca * y + sa * cx + (1 - ca) * cy
            x, y = nx, ny
        return (x + off_x + bx0, y + off_y + by0)

    centers = {}
    for r in range(n):
        for c in range(n):
            # 格中心 = 相邻两网格线的规整位置中点(不能用线位置本身 —
            # 那是格线交点, 点击会落在格线上被游戏随机判定到相邻格)
            cx = (v_fit[c] + v_fit[c + 1]) / 2.0
            cy = (h_fit[r] + h_fit[r + 1]) / 2.0
            centers[(r, c)] = to_input(cx, cy)
    board_origin = to_input(0.0, 0.0)      # 裁剪系原点 = 棋盘框左上角

    log.info("[识别] %s", Puzzle(n, trees, row_ct, col_ct, centers,
                                 step_x, step_y, board_origin,
                                 warnings).describe())
    return Puzzle(n, trees, row_ct, col_ct, centers, step_x, step_y,
                  board_origin, warnings, existing_tents=existing,
                  scale=scale, rot_ang=rot_ang)


def _estimate_cell(wall, board):
    """粗估格宽(尺度归一化判定用): 以最大投影峰间距近似."""
    h, w = wall.shape
    vp = wall.sum(axis=0)
    segs = _line_segs(vp, 0.55 * h)
    if len(segs) >= 3:
        steps = [b - a for (a, _e, _c), (b, _e2, _c2) in zip(segs[:-1],
                                                             segs[1:])]
        return float(np.median(steps))
    x0, _y0, x1, _y1 = board
    return float(x1 - x0) / 3.0


def find_board_bbox(img, cfg):
    """轻量锚点定位: 返回输入图坐标系中棋盘外接框 (x0,y0,x1,y1).

    供作答前校准使用(不识别内容, 只找网格外接框).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv, (off_x, off_y) = _crop_dark_border(hsv, cfg)
    _wall, (x0, y0, x1, y1) = _find_board(hsv, cfg)
    return (x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y)


def read_tent_cells(img, cfg):
    """回读棋盘上当前帐篷位置(提交前校验用).

    返回 (n, cells 集合); 定位失败返回 None. 只找网格+暗图标, 不读数字.
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv, (off_x, off_y) = _crop_dark_border(hsv, cfg)
        wall, _board = _find_board(hsv, cfg)        # wall 已裁剪, 直接用
        vsegs, hsegs, _cell = _grid_lines(wall, cfg)
        n = len(vsegs) - 1
        if n != len(hsegs) - 1:
            return None
        x0, y0, _x1, _y1 = _board
        hsv_c = hsv[y0:y0 + wall.shape[0], x0:x0 + wall.shape[1]]
        v_ch = hsv_c[:, :, 2]
        canopy = _canopy_mask(hsv_c)
        cells = set()
        for r in range(n):
            ya, yb = hsegs[r][1] + 1, hsegs[r + 1][0]
            for c in range(n):
                xa, xb = vsegs[c][1] + 1, vsegs[c + 1][0]
                if yb <= ya or xb <= xa:
                    continue
                if float(canopy[ya:yb, xa:xb].mean()) > 0.05:
                    continue                        # 树格, 非帐篷
                if float((v_ch[ya:yb, xa:xb] < 110).mean()) > \
                        float(cfg["tent_dark_frac"]):
                    cells.add((r, c))
        return n, cells
    except Exception:                             # 回读失败按不可用处理
        return None
