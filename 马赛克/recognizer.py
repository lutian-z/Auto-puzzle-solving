# -*- coding: utf-8 -*-
"""recognizer.py — 马赛克(Mosaic)题目图像识别(版面分析/网格提取/格状态/数字识别).

输入一张截图(用户粗框选区域, 无需像素级精确), 输出结构化题目 Puzzle:
    n / nums(数字矩阵, {(r,c): 0~9}) / black_cells(当前已涂黑格集合) /
    centers(每格中心像素坐标, 相对识别输入图) / board_bbox.

流程:
    1) 灰度三峰自适应: 素材为四级灰度画 — 黑格/数字墨≈0, 网格线≈102,
       格底≈204, 纸面≈255. 从直方图峰估计 v_black/v_line/v_bg 三个水平,
       一切二值化阈值都由它们推导, 亮度/对比度扰动下自动跟随(固定阈值
       在 ±25% 亮度/0.7~1.3 对比度下会把线或黑格吸错, 推演见 estimate_levels).
    2) 线色区间掩码 (黑+线)/2 < g < (线+底)/2: 只含网格线及其抗锯齿核心 —
       黑格(0)与格底(204)都在区间外. 这是与兄弟项目"暗像素"口径的关键
       差异: 答案图/对局中的大片黑格会毁掉暗掩码投影(整行全黑时投影满宽).
    3) 掩码最大连通域 = 网格线网络, 外接框即题目真实边界; 黑边预裁 →
       去倾斜(Hough 偏差中位数) → 尺度归一化(格 <22px 或 >72px 缩到 36px),
       全部映射回输入图坐标系.
    4) 投影法找网格线: 阈值 0.7×边长(线必然贯穿全宽/全高; 黑格镶边即使
       成峰也紧贴真线, 由近距合并吸收), 行/列线数必须相等(正方形),
       等差规整后残差受限. 几何失败时自动放宽线窗重试一次(小数缩放下
       网格线可被反锯齿抬到几乎与格底同灰), 两档都失败则抛原始错误.
    5) 逐格: 内部中值灰度判 涂黑/未涂; 内缩切片提取字形(未涂格深字/
       涂黑格白字, 按格状态选墨迹方向, 反状态兜底), 贴边组件(网格线
       残边)拒收, 归一化后与 templates.npz 灰度软字形模板做 相关性/IoU
       匹配. 低置信度触发二值化/形态学变体重认, 仍低则告警.
    6) 一致性校验: 数字范围 0..9 且 ≤ 邻域大小; 规模 ≥1 任意 N×N;
       自由格(不受任何数字约束)告警.

数字模板为 templates.npz 数据文件(系统多字体渲染 + 素材真实字形样本,
与代码同目录分发), 运行时不依赖字体与外部素材。
"""
import os

import cv2
import numpy as np

import solver

_TPL_FMT = 3          # 模板数据格式版本(灰度软字形+真实样本=3)
_TPL_SIZE = 32        # 模板归一化尺寸(与 _normalize_glyph 一致)
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "templates.npz")
_TPL_CACHE = None     # 模块级缓存, 避免每次识别重复读盘


# 异常体系(所有失败给出可读信息并可重试)

class RecognitionError(solver.MosaicError):
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
    interp = cv2.INTER_CUBIC if sc >= 1 else cv2.INTER_AREA
    g = cv2.resize(g, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = g
    return canvas.astype(np.float32) / 255.0


class DigitClassifier:
    """单字形 0-9 分类: 多模板 归一化互相关(灰度) + IoU(二值) 取最大,
    附次选分差. 灰度互相关对缩放/模糊的抗锯齿渐变更宽容, IoU 保底.

    性能关键实现(50×50 有 ~1200 字形 × ~180 模板, 逐对 matchTemplate
    实测 3.3s): 模板预堆叠为 (T, 32*32) 矩阵, 同尺寸归一化互相关即
    皮尔逊相关系数 corr = (Tc·ac)/(||Tc||·||ac||), 与全部模板的 IoU 的
    交集也是一次矩阵乘 — 每字形整分类化为两次 (T,1024)×(1024,) 矩阵乘,
    实测提速 ~30 倍, 分数与逐对实现数学等价(见 tests/test_recognizer)."""

    def __init__(self, templates):
        digits, tcs, tns, tbs, tbs_sum = [], [], [], [], []
        for d, tpls in templates.items():
            for t in tpls:
                tf = np.ascontiguousarray(t, dtype=np.float32).ravel()
                tc = tf - tf.mean()
                tn = float(np.linalg.norm(tc))
                tcs.append(tc)
                tns.append(tn if tn > 1e-9 else 1e-9)
                b = (t > 0.5).astype(np.float32).ravel()
                tbs.append(b)
                tbs_sum.append(float(b.sum()))
                digits.append(int(d))
        self._digits = np.array(digits, dtype=np.int64)
        self._tc = np.stack(tcs)                      # (T, 1024)
        self._tn = np.array(tns, dtype=np.float32)    # (T,)
        self._tb = np.stack(tbs)                      # (T, 1024)
        self._tbs = np.array(tbs_sum, dtype=np.float32)

    def classify(self, glyph_f32):
        """glyph_f32: 归一化 float32 掩码. 返回 (digit, score, margin)."""
        if glyph_f32 is None:
            return 0, 0.0, 0.0
        a = np.ascontiguousarray(glyph_f32, dtype=np.float32).ravel()
        ac = a - a.mean()
        an = float(np.linalg.norm(ac))
        if an < 1e-9:
            return 0, 0.0, 0.0
        corr = (self._tc @ ac) / (self._tn * an)      # (T,)
        ab = (a > 0.5).astype(np.float32)
        inter = self._tb @ ab                         # (T,)
        union = float(ab.sum()) + self._tbs - inter
        iou = np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)
        best_t = np.maximum(corr, iou)                # 每模板最高分
        per_digit = np.full(10, -1.0, dtype=np.float64)
        np.maximum.at(per_digit, self._digits, best_t)
        best_d = int(np.argmax(per_digit))
        order = np.sort(per_digit)[::-1]
        second = float(order[1]) if len(order) > 1 else 0.0
        return best_d, float(per_digit[best_d]), float(per_digit[best_d]) \
            - second


# 灰度三峰自适应与基础掩码

def _hist_peak(gray, lo, hi):
    """[lo, hi] 闭区间内最高峰的灰度值(区间无像素返回 None)."""
    hist = np.bincount(gray.ravel(), minlength=256)
    seg = hist[lo:hi + 1]
    if seg.sum() == 0:
        return None
    return lo + int(np.argmax(seg))


def estimate_levels(gray):
    """估计 (v_black, v_line, v_bg) 三个灰度水平.

    扰动推演(乘性亮度 a·v / 线性对比度 1.3·(v-128)+128 / 加性 +40):
      标准:        0 / 102 / 204 → (0, 102, 204)
      亮度x0.75:   0 /  76 / 153 → (0, 76, 153)
      亮度x1.25:   0 / 128 / 255 → (0, 128, 255)
      对比度0.7:  90 / 110 / 181 → (90, 110, 181)
      对比度1.3:   0 /  71 / 224 → (0, 71, 224)
      加性+40:    40 / 142 / 244 → (40, 142, 244)
    v_bg   = [120,255] 最高峰 (格底; x1.25 后与纸面同值仍命中)
    v_black= [0, 0.48·v_bg] 最高峰 (黑格/墨; 排除线峰 — 无黑格的题面图
             里线峰远大于墨峰, 上界必须把线挡在外面; 0.48·204≈97<102 ✓,
             0.48·255≈122<128 ✓, 0.48·181≈86<90 ✓, 0.48·224≈107>71 ✓)
    v_line = (v_black, v_bg) 之间扣除两端 15% 后的最高峰.
    """
    v_bg = _hist_peak(gray, 120, 255)
    if v_bg is None:
        v_bg = 204
    v_black = _hist_peak(gray, 0, int(0.48 * v_bg))
    if v_black is None:
        v_black = max(0, v_bg // 3 - 20)
    gap = int(0.15 * (v_bg - v_black))
    v_line = None
    if v_bg - v_black > 60:
        v_line = _hist_peak(gray, v_black + gap, v_bg - gap)
    if v_line is None:
        v_line = (v_black + v_bg) // 2
    return int(v_black), int(v_line), int(v_bg)


def _line_mask(gray, levels, widen=0):
    """网格线掩码: 灰度落在 (黑+线)/2 与 (线+底)/2 之间.

    只有线及其抗锯齿核心落在此区间; 黑格(0)与格底(204)都在区间外.
    黑格与格底交界的抗锯齿会短暂落入区间, 但只贴在黑格边缘、与真线
    连成一体, 不影响最大连通域与投影峰段. 中值去噪会抹断细线, 故不做.

    widen=1 只抬窗上界到 格底-max(4,3%格底), 救小数缩放下被反锯齿抬亮的
    网格线(实测下限: 线列距格底 >=6 灰阶; 再浅物理不可分, 宁可拒绝).
    仅供主窗口几何失败后的重试使用, widen=0 主路径逐位不变.
    """
    v_black, v_line, v_bg = levels
    lo = (v_black + v_line) // 2
    hi = (v_line + v_bg) // 2
    if widen == 1:
        hi = max(hi, v_bg - max(4, int(0.03 * v_bg)))
    return ((gray > lo) & (gray < hi)).astype(np.uint8)


def _largest_component(mask):
    """最大连通域(8邻). 返回 (组件掩码, 外接框 (x0,y0,x1,y1)) 或 None.

    闭运算前先剔除面积 <6 的孤立小组件: 椒盐噪声的点/短链会在闭运算中
    相互桥接并挂到网格组件上, 使外接框膨胀(帐篷项目实测教训).
    性能: 面积过滤用 stats 表按标签一次性查表清零(逐组件布尔索引在
    千级噪点组件下实测 1.5s, 查表 <50ms).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1 and (stats[1:, cv2.CC_STAT_AREA] < 6).any():
        small = np.zeros(n, dtype=bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < 6
        mask = mask * (~small[labels]).astype(np.uint8)
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
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


def _find_board(gray, levels, cfg, widen=0):
    """定位网格线网络组件与外接框(留 2px 边). 返回 (wall, bbox)."""
    h, w = gray.shape
    lm = _line_mask(gray, levels, widen=widen)
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


# 预处理(黑边裁剪/去倾斜/尺度归一化)

def _crop_dark_border(gray, cfg):
    """四周为深色背景时裁到最大亮色区域. 返回 (gray, (off_x, off_y))."""
    h, w = gray.shape
    thr = int(cfg["dark_v"])
    frame = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if (frame < thr).mean() < 0.6:
        return gray, (0, 0)
    bright = (gray >= thr).astype(np.uint8)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(bright, 4)
    if n <= 1:
        return gray, (0, 0)
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh = int(stats[i, 0]), int(stats[i, 1]), \
        int(stats[i, 2]), int(stats[i, 3])
    if bw < w * 0.6 or bh < h * 0.6:
        return gray, (0, 0)
    return gray[y:y + bh, x:x + bw], (x, y)


def _skew_deg_fine(gray, cfg):
    """Hough 直线偏差中位数估计倾斜角(度, 绝对值). 复制自数墙/帐篷项目."""
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


def _merge_close(segs, min_gap):
    """合并间距 < min_gap 的相邻峰段(黑格镶边偶发的贴邻峰), 重心按段长
    加权. 正常情形线峰唯一, 合并为空操作."""
    if len(segs) <= 1:
        return segs
    out = [list(segs[0])]
    for s, e, c in segs[1:]:
        if s - out[-1][1] < min_gap:
            w1 = out[-1][1] - out[-1][0] + 1
            w2 = e - s + 1
            out[-1][2] = (out[-1][2] * w1 + c * w2) / (w1 + w2)
            out[-1][1] = e
        else:
            out.append([s, e, c])
    return [(s, e, c) for s, e, c in out]


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

    阈值 0.7×边长: 网格线必然贯穿全宽/全高; 黑格镶边只覆盖黑格段,
    极端整行/列全黑时的贴邻峰由近距合并吸收(合并距离 ~0.4% 边长 ≈
    2~4px, 恰为镶边与真线的距离量级, 不会并掉相邻真线).
    """
    h, w = wall.shape
    vp = wall.sum(axis=0)
    hp = wall.sum(axis=1)
    gap = max(2, int(0.004 * max(h, w)))
    vsegs = _merge_close(_line_segs(vp, 0.7 * h), gap)
    hsegs = _merge_close(_line_segs(hp, 0.7 * w), gap)
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


# 逐格分析: 涂黑状态 + 数字识别

def _cell_ink(patch, state, levels, cfg):
    """单元格墨迹掩码(uint8 0/255) 或 None(墨迹占满, 状态可疑).

    主口径 = 逐格 Otsu: 对"该格实际的字色/底色"自适应分割, 尺度归一化
    (INTER_AREA)后墨迹被稀释成中间灰仍可分; 固定层级阈值此时会把字形
    啃蚀变形(实测 5→1/4→6 系统性误读). 未涂格取反相(深字=前景),
    涂黑格取正相(白字=前景).
    注意: 不做中值模糊 — 3×3 中值会整根抹掉小字号数字的 1px 竖笔
    ("1"/"7" 漏读的根因); 孤立噪点由组件面积阈值过滤.
    """
    if state == "gray":
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _t, ink = cv2.threshold(patch, 0, 255, flags)
    if (ink > 0).mean() > float(cfg["ink_frac_max"]):
        return None                        # 墨迹占满 → 状态判定可疑
    return ink.astype(np.uint8)


def _cell_ink_levels(patch, state, levels, cfg):
    """备选墨迹口径: 三峰层级阈值(噪声大时 Otsu 双峰失真后的保底)."""
    v_black, v_line, v_bg = levels
    if state == "gray":
        thr = max(30, (v_black + v_line) // 2)
        ink = (patch < thr).astype(np.uint8) * 255
    else:
        thr = min(235, (v_line + v_bg) // 2)
        ink = (patch > thr).astype(np.uint8) * 255
    if (ink > 0).mean() > float(cfg["ink_frac_max"]):
        return None
    return ink


def _glyph_components(ink, cfg):
    """从墨迹掩码提取字形组件, 返回 [(组件掩码uint8, x左缘), ...] 按左→右.

    贴边组件 = 网格线/镶边的抗锯齿残边(数字字形在格内部, 不贴切片边).
    多个非贴边组件时保留面积 ≥ 最大组件 25% 者(噪声碎片被滤掉);
    马赛克数字均为一位数, 正常恰一个组件.
    """
    if ink is None:
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    h_p, w_p = ink.shape
    comps = []
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < int(cfg["glyph_min_area"]):
            continue
        if x <= 0 or y <= 0 or x + bw >= w_p or y + bh >= h_p:
            continue                        # 贴边 = 线残边
        comps.append((stats[i, cv2.CC_STAT_AREA], x, i))
    if not comps:
        return []
    max_area = max(c[0] for c in comps)
    keep = [c for c in comps if c[0] >= 0.25 * max_area]
    keep.sort(key=lambda t: t[1])           # 按从左到右排序
    return [((labels == i).astype(np.uint8) * 255, x) for _a, x, i in keep]


def _classify_glyphs(comps, clf):
    """组件列表 → (首位数字, 置信度). 马赛克只有一位数, 取最左组件."""
    if not comps:
        return None, 0.0
    piece, _x = comps[0]
    g = _normalize_glyph(piece, _TPL_SIZE)
    d, score, _margin = clf.classify(g)
    return d, score


def _read_cell_digit(patch, state, levels, clf, cfg, where, warnings):
    """识别单元格数字. 返回 (value 或 None, conf, low).

    反状态兜底: 格状态误判时数字墨方向相反(深↔白), 反状态置信度显著
    更高才采信并告警. 变体重认(低置信度二次确认): Otsu 二值 / 轻度
    膨胀 / 轻度腐蚀 / 3x 放大后腐蚀 1~2 档(小字形放大插值后笔画偏粗,
    腐蚀逼近模板细笔画), 取各变体最高分结果; 仍低则告警。
    """
    best = (None, 0.0)
    for ink in (_cell_ink(patch, state, levels, cfg),
                _cell_ink_levels(patch, state, levels, cfg)):
        v, c = _classify_glyphs(_glyph_components(ink, cfg), clf)
        if c > best[1]:
            best = (v, c)
    value, conf = best
    if conf < float(cfg["glyph_conf"]):
        alt_state = "black" if state == "gray" else "gray"
        alt_best = (None, 0.0)
        for ink in (_cell_ink(patch, alt_state, levels, cfg),
                    _cell_ink_levels(patch, alt_state, levels, cfg)):
            v, c = _classify_glyphs(_glyph_components(ink, cfg), clf)
            if c > alt_best[1]:
                alt_best = (v, c)
        if alt_best[0] is not None and alt_best[1] > conf + 0.05:
            warnings.append(
                f"{where} 涂黑状态与字形方向矛盾, 已按反状态识别")
            value, conf = alt_best
    if conf >= float(cfg["glyph_conf"]) or value is None:
        low = value is not None and conf < float(cfg["glyph_conf"])
        if low:
            warnings.append(f"{where} 数字置信度低({conf:.2f}), 结果可能有误")
        return value, conf, low
    # 变体重认
    if state == "gray":
        _, otsu = cv2.threshold(patch, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, otsu = cv2.threshold(patch, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    work = otsu
    variants = []
    k = np.ones((2, 2), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    variants.append(cv2.dilate(work, k))
    variants.append(cv2.erode(work, k))
    try:
        big = cv2.resize(work, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
        variants.append(cv2.erode(big, k3))
        variants.append(cv2.erode(big, k3, iterations=2))
    except cv2.error:
        pass
    best = (value, conf)
    for vm in variants:
        try:
            comps = _glyph_components(vm, cfg)
            if not comps:
                continue
            d, score = _classify_glyphs(comps, clf)
            if score > best[1]:
                best = (d, score)
        except cv2.error:
            continue
    value, conf = best
    low = conf < float(cfg["glyph_conf"])
    if low:
        warnings.append(f"{where} 数字置信度低({conf:.2f}), 结果可能有误")
    return value, conf, low


# 主入口

class Puzzle:
    """识别结果: 题目结构 + 当前涂黑状态 + 几何."""

    def __init__(self, n, nums, black_cells, centers, cell_w, cell_h,
                 board_origin, warnings, scale=1.0, rot_ang=0.0):
        self.n = n
        self.nums = nums                    # {(r,c): int 0~9}
        self.black_cells = black_cells      # set[(r,c)] 当前已涂黑格
        self.centers = centers              # {(r,c): (x,y)} 相对识别输入图
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.board_origin = board_origin    # 棋盘外接框左上角(输入图坐标)
        self.warnings = warnings
        self.scale = scale                  # 预处理参数(校准可比性判断用)
        self.rot_ang = rot_ang

    def describe(self):
        return (f"{self.n}x{self.n}, {len(self.nums)} 个数字, "
                f"当前已涂黑 {len(self.black_cells)} 格")


def _recognize_cells(gray, board, v_fit, h_fit, levels, clf, cfg, warnings):
    """逐格提取: 涂黑状态 + 数字. 返回 (nums, black_cells).

    坐标系: v_fit/h_fit 是墙裁剪坐标系, 加 board 左上角映射回 gray.
    clf 传 None 时只回读涂黑状态不识别数字(提交前校验用).
    """
    bx0, by0 = board[0], board[1]
    n = len(v_fit) - 1
    inset = max(2, int(round(min(_fit_axis_step(v_fit), _fit_axis_step(h_fit))
                              * 0.12)))     # 避开线抗锯齿
    nums = {}
    black_cells = set()
    multi = 0                          # 一格多字形的格数(假粗格侦测)
    uneven = 0                         # 低均匀度格数(假粗格侦测)
    ch, cw = gray.shape
    v_black, v_line, _v_bg = levels
    for r in range(n):
        y0 = int(round(h_fit[r])) + inset
        y1 = int(round(h_fit[r + 1])) - inset
        for c in range(n):
            x0 = int(round(v_fit[c])) + inset
            x1 = int(round(v_fit[c + 1])) - inset
            if y1 - y0 < 4 or x1 - x0 < 4:
                raise GridGeometryError(
                    f"格({r + 1},{c + 1}) 内部区域异常, 请重新框选")
            patch = gray[by0 + y0:by0 + y1, bx0 + x0:bx0 + x1]
            med = float(np.median(patch))
            if med < (v_black + v_line) / 2.0:
                black_cells.add((r, c))
                state = "black"
            else:
                state = "gray"
            if clf is None:                # 纯状态回读(提交前校验用)
                continue
            # 假粗格侦测: 网格线大面积均匀漏检时整盘会被读成粗格假盘,
            # 与识别成败无关(假盘每格多数字反而识别失败). 真实格=单底色
            # +至多一字(多字形占比/低均匀度在两域零重叠), 命中即拒绝.
            comps = _glyph_components(_cell_ink(patch, state, levels, cfg),
                                      cfg)
            if len(comps) > 1:
                multi += 1
            if float(np.mean(np.abs(
                    patch.astype(np.int16) - np.median(patch)) <= 12)) \
                    < float(cfg.get("fake_grid_uniform", 0.62)):
                uneven += 1
            value, _conf, _low = _read_cell_digit(
                patch, state, levels, clf, cfg, f"格({r + 1},{c + 1})",
                warnings)
            if value is not None:
                nums[(r, c)] = int(value)
    if clf is not None:
        fake = (multi >= 2
                and multi > float(cfg.get("fake_grid_multi_frac", 0.3))
                * max(len(nums), multi))
        fake = fake or (uneven >= 2 and uneven >= 0.15 * n * n)
        if fake:
            raise GridGeometryError(
                "格内容与网格规模矛盾(一格多字/格内混装子格), 网格线疑似"
                "大面积漏检被误读成粗格假盘, 请放大题目或重新框选")
    return nums, black_cells


def _fit_axis_step(fit):
    """规整线位置的相邻平均间距."""
    if len(fit) < 2:
        return 36.0
    return float((fit[-1] - fit[0]) / (len(fit) - 1))


def _locate_grid(gray, cfg, widen, notes):
    """步骤3~4: 三峰灰度+定位棋盘+尺度归一化+网格线投影+等差规整.

    notes 收集本阶段告警(调用方决定并入方式); widen 透传给线掩码。
    返回 (gray, levels, board, scale, v_fit, h_fit, step_x, step_y)。
    """
    levels = estimate_levels(gray)
    wall, board = _find_board(gray, levels, cfg, widen=widen)
    cell0 = _estimate_cell(wall, board)
    scale = 1.0
    if 0 < cell0 and (cell0 < 22.0 or cell0 > 72.0):
        scale = max(0.4, min(2.5, 36.0 / cell0))
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1
                          else cv2.INTER_AREA)
        wall = cv2.resize(wall.astype(np.uint8), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        board = tuple(int(round(v * scale)) for v in board)
        levels = estimate_levels(gray)
        notes.append(
            f"单元格 {cell0:.0f}px 偏{'小' if scale > 1 else '大'}, "
            f"已按 {scale:.2f}x 归一化尺度")

    # 网格线
    vsegs, hsegs, cell = _grid_lines(wall, cfg)
    n = len(vsegs) - 1
    if not (1 <= n <= int(cfg["max_board"])):
        raise GridGeometryError(
            f"网格规模异常(N={n}), 超出支持范围(1~{cfg['max_board']})")
    v_fit, step_x, res_x = _fit_axis([c for _s, _e, c in vsegs], cell)
    h_fit, step_y, res_y = _fit_axis([c for _s, _e, c in hsegs], cell)
    tol = 0.25 * max(step_x, step_y)
    if res_x > tol or res_y > tol:
        notes.append(
            f"网格线位置残差偏大(x={res_x:.1f}px, y={res_y:.1f}px), "
            f"已按等差规整")
    return gray, levels, board, scale, v_fit, h_fit, step_x, step_y


def recognize(img, cfg, log):
    """识别整道马赛克题目. img: BGR ndarray. 失败抛 RecognitionError 系."""
    if img is None or img.size == 0:
        raise RecognitionError("输入图像为空")
    warnings = []
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray0.shape

    # 1) 黑边预裁
    gray, (off_x, off_y) = _crop_dark_border(gray0, cfg)
    if off_x or off_y:
        warnings.append("检测到四周深色边框, 已自动裁剪到纸面区域")
    h1, w1 = gray.shape

    # 2) 去倾斜(迭代≤2轮: Hough 角度量化 0.25°, 大棋盘上残留会
    #      漂移投影; 多次重采样损失字形精度, 轮数须受限) ----
    rot_ang = 0.0
    for _it in range(2):
        ang0 = _skew_deg_fine(gray, cfg)
        if ang0 < float(cfg["skew_warn_deg"]):
            break
        cand = _rotate_gray(gray, ang0)
        ang1 = _skew_deg_fine(cand, cfg)
        if ang1 > ang0:                       # 方向反了: 反向转
            cand = _rotate_gray(gray, -ang0)
            rot_ang -= ang0
        else:
            rot_ang += ang0
        gray = cand
        warnings.append(f"题目存在 {abs(ang0):.1f}° 倾斜, 已自动转正")
    if abs(rot_ang) > float(cfg["skew_max_deg"]):
        raise GridGeometryError(
            f"题目倾斜约 {abs(rot_ang):.1f}° 超过上限 {cfg['skew_max_deg']}°, "
            f"请摆正窗口后重新框选")
    h2, w2 = gray.shape

    # 3~6) 定位棋盘 + 尺度归一化 + 网格线 + 逐格识别 + 一致性校验
    # 主窗口失败时自动放宽线窗重试一次(只多试不改道); 任何 RecognitionError
    # 都触发重试(漏线步长恰均匀时会误判成粗格假盘, 到数字一致性才穿帮)。
    # 两条路都失败则抛首次的原始错误。
    clf = DigitClassifier(_load_templates())
    notes = []
    err0 = None
    res = None
    for widen in (0, 1):
        try:
            (g_try, levels, board, scale,
             v_fit, h_fit, step_x, step_y) = _locate_grid(gray, cfg,
                                                          widen, notes)
            n = len(v_fit) - 1
            nums, black_cells = _recognize_cells(
                g_try, board, v_fit, h_fit, levels, clf, cfg, notes)
            if not nums:
                raise RecognitionError(
                    "未在棋盘内识别到任何数字, 请确认框选的是马赛克题目")
            for (r, c), k in sorted(nums.items()):
                size = len(solver.neighborhood(n, r, c))
                if not (0 <= k <= size):
                    raise RecognitionError(
                        f"格({r + 1},{c + 1}) 数字 {k} 超出邻域大小 {size}, "
                        f"数字识别可能有误, 请重新框选")
            covered = set()
            for _pos, _k, cells_ in solver.build_constraints(n, nums):
                covered.update(cells_)
            free_cnt = n * n - len(covered)
            if free_cnt:
                notes.append(
                    f"{free_cnt} 个格子不受任何数字约束(题目将无法唯一求解), "
                    f"请确认框选完整")
            res = (g_try, board, scale, v_fit, h_fit,
                   step_x, step_y, n, nums, black_cells)
        except RecognitionError as e:
            if err0 is None:
                err0 = e
            notes = []
        else:
            if widen:
                notes.insert(0, "网格线被反锯齿抬亮至接近格底, "
                                "已放宽线色窗口重试成功")
            warnings.extend(notes)
            break
    if res is None:
        raise err0
    (gray, board, scale, v_fit, h_fit,
     step_x, step_y, n, nums, black_cells) = res

    # 坐标回溯映射到输入图
    # 线位置/格中心在"墙裁剪坐标系"(裁剪原点=棋盘外接框左上角 board[0:2]),
    # 输出的 centers/board_origin 必须加回裁剪原点再映射到输入图坐标系
    # (帐篷项目实测教训: 漏掉此步会整体偏移一格).
    def to_input(x, y):
        if scale != 1.0:
            x, y = x / scale, y / scale
        if rot_ang:
            # warpAffine(gray, getRotationMatrix2D(c, θ)) 的内容移动是
            # q = M(θ)·p(实测往返验证), 因此"转正图→原图"必须用 M(-θ)。
            # 旧实现误用 M(+θ)(=再正向旋一次, 偏差 2θ·半径, 3° 时可达
            # 38px); 该分支仅在检测到 ≥0.7° 倾斜时执行, 常规截图不走。
            a = -np.deg2rad(rot_ang)
            ca, sa = np.cos(a), np.sin(a)
            cx_, cy_ = w2 / 2.0, h2 / 2.0
            nx = ca * x + sa * y + (1 - ca) * cx_ - sa * cy_
            ny = -sa * x + ca * y + sa * cx_ + (1 - ca) * cy_
            x, y = nx, ny
        return (x + off_x, y + off_y)

    centers = {}
    for r in range(n):
        for c in range(n):
            # 格中心 = 相邻两网格线规整位置的中点(不能用线交点 — 点击会
            # 落在格线上被游戏判定到相邻格)
            cx = (v_fit[c] + v_fit[c + 1]) / 2.0 + board[0]
            cy = (h_fit[r] + h_fit[r + 1]) / 2.0 + board[1]
            centers[(r, c)] = to_input(cx, cy)
    board_origin = to_input(board[0], board[1])

    log.info("[识别] %s", Puzzle(n, nums, black_cells, centers,
                                 step_x / scale, step_y / scale,
                                 board_origin, warnings).describe())
    return Puzzle(n, nums, black_cells, centers, step_x / scale,
                  step_y / scale, board_origin, warnings,
                  scale=scale, rot_ang=rot_ang)


def _find_board_retry(gray, levels, cfg):
    """定位棋盘: 主窗口失败时用放宽窗重试一次(锚点/回读共用, 同 recognize)."""
    try:
        return _find_board(gray, levels, cfg)
    except GridGeometryError as e0:
        try:
            return _find_board(gray, levels, cfg, widen=1)
        except GridGeometryError:
            raise e0


def find_board_bbox(img, cfg):
    """轻量锚点定位: 返回输入图坐标系中棋盘外接框 (x0,y0,x1,y1).

    供作答前校准使用(不识别内容, 只找网格外接框).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray, (off_x, off_y) = _crop_dark_border(gray, cfg)
    levels = estimate_levels(gray)
    _wall, (x0, y0, x1, y1) = _find_board_retry(gray, levels, cfg)
    return (x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y)


def read_black_cells(img, cfg):
    """回读棋盘上当前已涂黑格(提交前校验用).

    返回 (n, cells 集合); 定位失败返回 None. 只找网格+格底状态, 不读数字.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray, (off_x, _off_y) = _crop_dark_border(gray, cfg)
        levels = estimate_levels(gray)
        try:
            wall, board = _find_board(gray, levels, cfg)
            vsegs, hsegs, cell = _grid_lines(wall, cfg)
        except GridGeometryError as e0:
            try:
                wall, board = _find_board(gray, levels, cfg, widen=1)
                vsegs, hsegs, cell = _grid_lines(wall, cfg)
            except GridGeometryError:
                raise e0
        n = len(vsegs) - 1
        if n != len(hsegs) - 1:
            return None
        v_fit, _sx, _rx = _fit_axis([c for _s, _e, c in vsegs], cell)
        h_fit, _sy, _ry = _fit_axis([c for _s, _e, c in hsegs], cell)
        nums, cells = _recognize_cells(gray, board, v_fit, h_fit,
                                           levels, None, cfg, [])
        return n, cells
    except Exception:                         # 回读失败按不可用处理
        return None
