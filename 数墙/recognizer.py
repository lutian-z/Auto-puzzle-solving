# -*- coding: utf-8 -*-
"""recognizer.py — 数墙题目图像识别(版面分析/网格提取/任意形状直通/数字识别).

输入一张截图(用户粗框选区域, 无需像素级精确), 输出结构化棋盘:
    Puzzle: a×b 存在矩阵(shape) + 数字字典(nums) + 每格中心像素坐标.

流程:
    1. 暗像素掩码 → 形态学闭运算 → 取最大连通域作为"墙体"(网格线网络);
       墙体外接框即为题目真实边界(异形题自动贴合).
    2. 墙体补集的连通域 = 各存在格的内部区域(数字墨迹不属于墙体, 归入内部);
       用内部组件质心聚类出所有行/列 → a、b 与每格几何, 顺带得到形状.
    3. 存在矩阵直通求解与作答, 任意形状无需模板 (shape_params 仅作描述).
    4. 每个存在格: 内部 Otsu → 连通域 → 按水平序分组字形 → 归一化后
       与多字体渲染的 0-9 模板做相关性/IoU 匹配 → 1~3 位数逐位识别.
    5. 低置信度/模糊匹配打印 WARN; 整体失败抛 RecognizeError(入口层给出重试提示).

字体模板在多平台候选字体中搜索渲染(路径仅经环境锚点动态探测, 不写死),
并缓存到可配置的 npz; 识别精度不依赖特定字体, 任意 1~3 位数均可泛化.
"""
import os

import cv2
import numpy as np

import solver

IS_WINDOWS = sys_platform_windows = os.name == "nt"


class RecognizeError(Exception):
    """识别失败(可重试), 带用户可读信息."""


# 字体探测与数字模板

def _font_candidates():
    """跨平台等宽无衬线粗体/常规体候选, 全部经环境锚点动态探测."""
    cands = []
    if IS_WINDOWS:
        windir = os.environ.get("WINDIR", r"C:\Windows")
        fdir = os.path.join(windir, "Fonts")
        for name in ["verdanab.ttf", "verdana.ttf", "arialbd.ttf", "arial.ttf",
                     "tahoma.ttf", "segoeuib.ttf", "segoeui.ttf",
                     "calibrib.ttf", "calibri.ttf", "msyhbd.ttc", "msyh.ttc"]:
            cands.append(os.path.join(fdir, name))
    else:
        roots = []
        for env in ("XDG_DATA_DIRS",):
            for d in (os.environ.get(env) or "").split(":"):
                if d:
                    roots.append(os.path.join(d, "fonts"))
        roots += [os.path.join("/usr", "share", "fonts"),
                  os.path.expanduser(os.path.join("~", ".fonts")),
                  os.path.join("/Library", "Fonts")]
        names = ["truetype/dejavu/DejaVuSans-Bold.ttf",
                 "truetype/dejavu/DejaVuSans.ttf",
                 "truetype/liberation/LiberationSans-Bold.ttf",
                 "truetype/liberation/LiberationSans-Regular.ttf",
                 "truetype/liberation2/LiberationSans-Bold.ttf",
                 "truetype/lato/Lato-Bold.ttf",
                 "truetype/lato/Lato-Regular.ttf",
                 "truetype/noto/NotoSans-Bold.ttf",
                 "truetype/noto/NotoSans-Regular.ttf",
                 "truetype/freefont/FreeSansBold.ttf"]
        for root in roots:
            for n in names:
                cands.append(os.path.join(root, n))
    return [p for p in cands if os.path.exists(p)]


def _render_glyph(ch, font_path, size):
    """用指定字体渲染字符并归一化为 size×size 掩码(float32 0..1)."""
    from PIL import Image, ImageDraw, ImageFont
    canvas = size * 3
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size * 2)
    # 基线放在画布中下方, 保证上伸部与下伸部都完整落在画布内
    draw.text((size // 2, size * 2), ch, fill=255, font=font, anchor="ls")
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return None
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return _normalize_glyph((arr > 100).astype(np.uint8) * 255, size)


def _normalize_glyph(mask_u8, size):
    """把二值字形掩码等比缩放并居中到 size×size 画布, 返回 float32 0..1."""
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return None
    g = mask_u8[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    sc = (size - 6) / max(h, w)
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = g
    return canvas.astype(np.float32) / 255.0


def build_templates(cfg):
    """构建 0-9 数字模板 {digit: [float32 掩码]}.

    缓存为项目目录下固定文件 templates.npz: 首次运行生成并保存,
    之后永久复用; 文件缺失/损坏时自动重建.
    """
    size = int(cfg["template_size"])
    fonts = _font_candidates()
    cache_path = cfg.get("template_cache") or ""
    if cache_path and cfg.get("use_template_cache", True) \
            and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            tpls = {}
            for k in data.files:
                d = int(k.split("_")[0])
                tpls.setdefault(d, []).append(data[k].astype(np.float32))
            if all(d in tpls for d in range(10)):
                print(f"[模板] 从缓存加载: {os.path.basename(cache_path)}")
                return tpls
        except Exception:
            pass
    if not fonts:
        raise RecognizeError(
            "未找到可用系统字体, 无法生成数字模板; "
            "请安装常见字体或将模板缓存 npz 放到项目目录")
    from PIL import ImageFont  # noqa: F401  (确保 Pillow 可用再开始渲染)
    tpls = {d: [] for d in range(10)}
    used = 0
    for fp in fonts:
        try:
            ok = True
            glyph_map = {}
            for d in range(10):
                g = _render_glyph(str(d), fp, size)
                if g is None:
                    ok = False
                    break
                glyph_map[d] = g
            if not ok:
                continue
            for d in range(10):
                tpls[d].append(glyph_map[d])
            used += 1
        except Exception:
            continue
    if used == 0:
        raise RecognizeError("数字模板渲染失败: 无有效字体")
    print(f"[模板] 用 {used} 种字体渲染 0-9 模板(每种 {size} 尺寸)")
    if cache_path and cfg.get("use_template_cache", True):
        try:
            arrs = {}
            for d in range(10):
                for i, g in enumerate(tpls[d]):
                    arrs[f"{d}_{i}"] = g
            np.savez(cache_path, **arrs)
            print(f"[模板] 缓存到 {os.path.basename(cache_path)}")
        except Exception:
            pass
    return tpls


class DigitClassifier:
    """单字形 0-9 分类: 多字体模板 相关性+IoU 取最大, 附次选分差."""

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
                res = cv2.matchTemplate(glyph_f32, t, cv2.TM_CCOEFF_NORMED)
                corr = float(res.max())
                b = t > 0.5
                union = (a | b).sum()
                iou = float((a & b).sum()) / union if union else 0.0
                best = max(best, corr, iou)
            scores[d] = best
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        best_d, best_s = order[0]
        second = order[1][1] if len(order) > 1 else 0.0
        return best_d, best_s, best_s - second


# 版面分析

def _largest_components(mask, keep=4):
    """返回掩码中最大的 keep 个连通域(按面积降序)的布尔掩码列表."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return []
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    out = []
    for i in order[:keep]:
        m = labels == i
        out.append(m)
    return out


def _wall_and_crop(gray, cfg):
    """找墙体组件并裁到其外接框(留 2px 边). 返回 (crop_gray, wall, (x0,y0)).

    主阈值失败时自动用"格体众数×0.87"的相对阈值重试一次(救浏览器小数
    缩放下被反锯齿抬亮的网格线); 仍失败则报原始错误.
    """
    try:
        return _wall_and_crop_thr(gray, cfg, int(cfg["dark_threshold"]))
    except RecognizeError as err0:
        body = int(np.argmax(np.bincount(np.asarray(gray).ravel())))
        thr_rel = max(int(cfg["dark_threshold"]), min(240, int(body * 0.87)))
        if thr_rel == int(cfg["dark_threshold"]):
            raise err0
        try:
            return _wall_and_crop_thr(gray, cfg, thr_rel)
        except RecognizeError:
            raise err0


def _wall_and_crop_thr(gray, cfg, dark_thr):
    dark = (gray < dark_thr).astype(np.uint8)
    k = max(1, int(cfg["morph_close_size"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    h, w = gray.shape
    min_area = max(200.0, float(cfg["wall_min_frac"]) * h * w)
    for cand in _largest_components(closed, keep=4):
        area = cand.sum()
        if area < min_area:
            break
        ys, xs = np.where(cand)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if bw < 3 * int(cfg["min_cell_px"]) or bh < 3 * int(cfg["min_cell_px"]):
            continue
        if not (0.35 <= bw / float(bh) <= 2.8):
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        m = 2
        x0c, y0c = max(0, x0 - m), max(0, y0 - m)
        x1c, y1c = min(w, x1 + m), min(h, y1 + m)
        wall = cand[y0c:y1c, x0c:x1c]
        return gray[y0c:y1c, x0c:x1c], wall, (x0c, y0c)
    raise RecognizeError(
        "未能在框选区域内定位题目网格(墙体). "
        "请框选时完整包含整道题目, 并避免混入其他深色界面元素")


def _skew_deg(wall):
    """用墙体最小外接矩形估计倾斜角(度, 取绝对值, 归一到 0~45)."""
    ys, xs = np.where(wall)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) < 50:
        return 0.0
    (_, _), (_, _), ang = cv2.minAreaRect(pts)
    ang = abs(ang)
    if ang > 45:
        ang = 90 - ang
    return ang


def _skew_deg_fine(gray, cfg):
    """Hough 直线偏差中位数估计倾斜角(度, 绝对值).

    对图像中的长直线(网格线)取与最近坐标轴的角度偏差的中位数 —
    比 minAreaRect 稳健: 不依赖墙体点集分布, 对亮度变化/噪声不敏感,
    角度分辨率 0.25°.
    """
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120,
                            minLineLength=max(60, int(gray.shape[1] * 0.15)),
                            maxLineGap=6)
    if lines is None:
        return 0.0
    devs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        if a <= 45:
            devs.append(a)                 # 近水平
        elif a >= 135:
            devs.append(180.0 - a)         # 近水平(另一象限)
        else:
            devs.append(abs(a - 90.0))     # 近垂直
    if not devs:
        return 0.0
    return float(np.median(devs))


def _rotate_gray(gray, ang_deg):
    """绕中心旋转(复制填充, 不引入黑角)."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _line_wall(wall, cell_est):
    """从墙体组件中提取"纯网格线"掩码(形态学长核开运算).

    有的渲染字号大到数字与网格线相连, 数字会并入墙体最大连通域;
    但数字笔画/字宽总是明显短于网格线段, 用长度 ≥ 1.2×格宽的
    水平/垂直核做开运算即可只保留线条、剔除数字墨迹.
    """
    L = max(9, int(1.2 * cell_est))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    h_lines = cv2.morphologyEx(wall, cv2.MORPH_OPEN, kh)
    v_lines = cv2.morphologyEx(wall, cv2.MORPH_OPEN, kv)
    lines = cv2.bitwise_or(h_lines, v_lines)
    return lines


def _interior_cells(wall, cfg):
    """墙体补集的连通域 → 候选格内部组件.

    返回 (comps, med_area): comps 为 [(cx, cy, area)], 已按面积过滤.
    """
    light = (~wall).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(light, 8)
    areas = []
    raw = []
    hh, ww = wall.shape
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        touches = (x <= 0 or y <= 0 or x + bw >= ww or y + bh >= hh)
        if touches:
            continue                       # 外部背景(贴裁剪边)
        raw.append((centroids[i][0], centroids[i][1],
                    float(stats[i, cv2.CC_STAT_AREA])))
        areas.append(float(stats[i, cv2.CC_STAT_AREA]))
    if not raw:
        raise RecognizeError("题目内部未检测到任何单元格区域")
    med = float(np.median(areas))
    comps = [t for t in raw
             if cfg["cell_area_lo"] * med <= t[2] <= cfg["cell_area_hi"] * med]
    return comps, med


def _cluster_1d(values, gap):
    """一维聚类: 排序后按间隔 > gap 分组, 返回各组均值(升序)."""
    vals = sorted(values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [float(np.mean(g)) for g in groups]


def _fit_axis(centers, cell_est):
    """校验某轴格中心近似等差并做最小二乘规整. 返回 (positions, step, residual)."""
    pos = np.array(centers, dtype=np.float64)
    idx = np.arange(len(pos), dtype=np.float64)
    n = len(idx)
    if n < 2:
        return pos.tolist(), float(cell_est), 0.0
    sx, sy = idx.sum(), pos.sum()
    sxx = (idx * idx).sum()
    sxy = (idx * pos).sum()
    denom = n * sxx - sx * sx
    step = ((n * sxy - sx * sy) / denom) if abs(denom) > 1e-9 else float(cell_est)
    intercept = (sy - step * sx) / n
    fitted = intercept + step * idx
    residual = float(np.abs(pos - fitted).max())
    return fitted.tolist(), float(step), residual


def _layout_from_components(comps, med_area, cfg):
    """由格内部组件质心推断 (rows_y, cols_x, cell_size)."""
    cell_est = float(np.sqrt(med_area))
    gap = 0.4 * cell_est
    rows_y = _cluster_1d([c[1] for c in comps], gap)
    cols_x = _cluster_1d([c[0] for c in comps], gap)
    a, b = len(rows_y), len(cols_x)
    max_board = int(cfg["max_board"])
    if not (2 <= a <= max_board and 2 <= b <= max_board):
        raise RecognizeError(
            f"网格规模异常(行={a}, 列={b}), 框选区域可能不完整或混入了其他内容")
    ry, step_y, res_y = _fit_axis(rows_y, cell_est)
    rx, step_x, res_x = _fit_axis(cols_x, cell_est)
    tol = 0.25 * max(step_y, step_x)
    if res_y > tol or res_x > tol:
        raise RecognizeError(
            f"网格线规整度不足(残差 y={res_y:.1f}px, x={res_x:.1f}px > {tol:.1f}px), "
            f"题目可能倾斜或被遮挡; 请重新框选")
    return ry, rx, step_y, step_x, cell_est


def _assign_cells(comps, rows_y, cols_x):
    """把组件质心就近指派到 (row, col), 返回 a×b 存在矩阵与冲突列表."""
    a, b = len(rows_y), len(cols_x)
    grid = [[False] * b for _ in range(a)]
    conflict = 0
    for (cx, cy, _area) in comps:
        r = int(np.argmin([abs(cy - y) for y in rows_y]))
        c = int(np.argmin([abs(cx - x) for x in cols_x]))
        if grid[r][c]:
            conflict += 1
        grid[r][c] = True
    return grid, conflict


def _shape_connected(exist):
    """存在格四邻域是否彼此连通(空盘视为不连通, 由上游保证不会发生)."""
    a = len(exist)
    b = len(exist[0]) if a else 0
    total = sum(sum(1 for v in row if v) for row in exist)
    if total == 0:
        return False
    start = next((r, c) for r in range(a) for c in range(b) if exist[r][c])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if (0 <= p[0] < a and 0 <= p[1] < b and exist[p[0]][p[1]]
                    and p not in seen):
                seen.add(p)
                stack.append(p)
    return len(seen) == total


# 字形提取与数值识别

def _cell_glyphs(cell_gray, cfg, wall_patch=None):
    """从单元格内部图提取字形掩码列表(按从左到右排序). 无数字返回 [].

    wall_patch: 与 cell_gray 同尺寸的墙体掩码(bool), 用于抹掉网格线残边,
    避免边框被 Otsu 并入字形导致误识别.
    """
    if cell_gray.size == 0:
        return []
    _, otsu = cv2.threshold(cell_gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_g = (cell_gray < int(cfg["dark_threshold"])).astype(np.uint8) * 255
    # 两者取墨水较多者作为掩码(抗背景偏色)
    mask = otsu if (otsu > 0).sum() >= (dark_g > 0).sum() else dark_g
    if wall_patch is not None:
        mask = mask & (~wall_patch)
    ink_frac = (mask > 0).mean()
    if ink_frac < float(cfg["cell_ink_min_frac"]):
        return []
    # 去噪: 只保留面积达标的组件. 网格线已由墙体掩码剔除, 故不再做
    # 贴边拒判 —— 有的渲染字体几乎占满整格(字形被网格线轻微切入),
    # 贴边拒判会把整个数字丢掉; 残余线碎片面积很小, 由面积阈值过滤.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < int(cfg["glyph_min_area"]):
            continue
        comps.append((x, y, bw, bh, area, i))
    if not comps:
        return []
    min_h = min(c[3] for c in comps)
    out = []
    for (x, y, bw, bh, area, i) in comps:
        piece = (labels == i).astype(np.uint8) * 255
        if bw > 1.55 * max(min_h, bh):     # 疑似两数字粘连 → 按竖直谷切开
            out.extend(_split_wide(piece))
        else:
            out.append(piece)
    return out


def _split_wide(mask):
    """把疑似粘连的宽组件按列墨水谷分割为若干字形."""
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
        if (piece > 0).sum() > 0:
            out.append(piece)
    return out if out else [mask]


def _read_cell_value(cell_gray, clf, cfg, where, wall_patch=None):
    """识别单元格数字. 返回 (value, conf, low_conf). 空格返回 (0, 0, False)."""
    glyphs = _cell_glyphs(cell_gray, cfg, wall_patch)
    if not glyphs:
        return 0, 0.0, False
    if len(glyphs) > 3:
        print(f"[识别] {where} 字形数异常({len(glyphs)}), 仅取前 3 个")
        glyphs = glyphs[:3]
    size = int(cfg["template_size"])
    digits = []
    confs = []
    low = False
    for gi, gmask in enumerate(glyphs):
        g = _normalize_glyph(gmask, size)
        d, score, margin = clf.classify(g)
        if d == 0 or score < float(cfg["glyph_conf"]):
            low = True
            print(f"[识别] {where} 第{gi + 1}位数字置信度低: 匹配 {d} 分 "
                  f"{score:.2f} (阈值 {float(cfg['glyph_conf']):.2f}), "
                  f"结果可能有误")
        elif margin < float(cfg["glyph_margin_ratio"]):
            print(f"[识别] {where} 第{gi + 1}位数字区分度低: "
                  f"{d}({score:.2f}) 与次选差 {margin:.2f}")
        digits.append(d)
        confs.append(score)
    value = int("".join(str(d) for d in digits))
    return value, float(min(confs)) if confs else 0.0, low


# 主入口

class Puzzle:
    """识别结果: 形状 + 数字 + 几何."""

    def __init__(self, a, b, shape, nums, centers, cell_w, cell_h,
                 crop_origin, params, warnings):
        self.a = a                      # 行数
        self.b = b                      # 列数
        self.shape = shape              # a×b bool, True=格存在
        self.nums = nums                # {(r,c): int}
        self.centers = centers          # {(r,c): (x,y)} 相对识别输入图
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.crop_origin = crop_origin  # 墙体外接框在输入图中的左上角
        self.params = params            # solver.shape_params 结果(自由形状为 None)
        self.warnings = warnings

    def describe(self):
        if self.params is None:
            n = sum(sum(1 for v in row if v) for row in self.shape)
            return (f"{self.a}x{self.b}, 自由形状(存在 {n} 格), "
                    f"{len(self.nums)} 个数字")
        a, b, corners, middle = self.params
        cdesc = "/".join(f"{r}x{c}" for (r, c) in corners)
        mdesc = "-" if middle is None else (
            f"@({middle[0]},{middle[1]}) {middle[2]}x{middle[3]}")
        return (f"{a}x{b}, 四角裁[{cdesc}], 中间裁{mdesc}, "
                f"{len(self.nums)} 个数字")


def _crop_dark_border(gray, cfg):
    """四周为深色背景(如桌面/黑边)时, 裁到最大亮色(纸面)区域.

    黑边会与最外圈格线粘连成环, 破坏墙体组件与形状拟合. 判定条件:
    图像最外圈一圈像素大部分深于阈值才处理; 裁剪结果过小(<60%)则放弃.
    """
    h, w = gray.shape
    frame = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if (frame < cfg["dark_threshold"]).mean() < 0.6:
        return gray
    bright = (gray >= cfg["dark_threshold"]).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(bright, 4)
    if n <= 1:
        return gray
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh = int(stats[i, 0]), int(stats[i, 1]), \
        int(stats[i, 2]), int(stats[i, 3])
    if bw < w * 0.6 or bh < h * 0.6:
        return gray
    return gray[y:y + bh, x:x + bw]


def recognize(img, cfg, clf=None):
    """识别整道数墙题目. img: BGR ndarray. 失败抛 RecognizeError.

    预处理(坐标可回溯映射, Puzzle.centers/crop_origin 始终相对输入图):
      1) 黑边预裁: 四周深色背景先裁到纸面, 避免黑边与格线粘连;
      2) 去倾斜: 检出 0.15°~skew_max_deg 的轻微倾斜先旋转转正;
      3) 尺度归一化: 单元格 <24px 或 >64px 时整体缩放到 ~36px,
         使形态学核/面积阈值等尺度相关参数保持稳健.
    """
    if img is None or img.size == 0:
        raise RecognizeError("输入图像为空")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warnings = []

    # 1) 黑边预裁
    h0, w0 = gray.shape
    off_x = off_y = 0
    gray_c = _crop_dark_border(gray, cfg)
    if gray_c.shape != gray.shape:
        # 裁剪偏移 = 亮区在原图中的位置
        bright = (gray >= cfg["dark_threshold"]).astype(np.uint8)
        n_, _l, stats, _c = cv2.connectedComponentsWithStats(bright, 4)
        i = 1 + int(np.argmax(stats[1:, 4]))
        off_x, off_y = int(stats[i, 0]), int(stats[i, 1])
        warnings.append("检测到四周深色边框, 已自动裁剪到纸面区域")
        gray = gray_c
    h1, w1 = gray.shape

    # 2) 去倾斜(一次性, 方向自动校正)
    # 轻微倾斜(< skew_warn_deg)由轴对齐拟合天然容忍, 不旋转以免插值
    # 损伤字形; 达到告警阈值才转正.
    rot_ang = 0.0
    try:
        ang0 = _skew_deg_fine(gray, cfg)
    except Exception:
        ang0 = 0.0
    if float(cfg["skew_warn_deg"]) <= ang0 <= float(cfg["skew_max_deg"]):
        gray_try = _rotate_gray(gray, ang0)
        try:
            ang1 = _skew_deg_fine(gray_try, cfg)
        except Exception:
            ang1 = 0.0
        if ang1 <= ang0:
            rot_ang = ang0               # 方向正确
        else:                            # 方向反了: 反向转
            gray_try = _rotate_gray(gray, -ang0)
            rot_ang = -ang0
        gray = gray_try
        warnings.append(f"题目存在 {abs(rot_ang):.1f}° 倾斜, 已自动转正")

    # 3) 尺度归一化(保守: 仅明显过小/过大才触发, 原生尺度附近
    #      不动, 以免插值与阈值漂移引入识别抖动) ----
    scale = 1.0
    try:
        _c1, wall1, _o1 = _wall_and_crop(gray, cfg)
        _comps, med = _interior_cells(wall1, cfg)
        cell0 = float(np.sqrt(med))
    except RecognizeError:
        cell0 = 0.0
    if 0 < cell0 and (cell0 < 22.0 or cell0 > 72.0):
        scale = max(0.4, min(2.5, 36.0 / cell0))
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1
                          else cv2.INTER_LINEAR)
        warnings.append(
            f"单元格 {cell0:.0f}px 偏{'小' if scale > 1 else '大'}, "
            f"已按 {scale:.2f}x 归一化尺度")

    # 闭运算核递增重试, 提升对断线/粘连的鲁棒性
    close_sizes = [int(cfg["morph_close_size"]),
                   int(cfg["morph_close_size"]) + 2,
                   int(cfg["morph_close_size"]) + 4]
    last_err = None
    puz = None
    for k in close_sizes:
        cfg_try = dict(cfg)
        cfg_try["morph_close_size"] = k
        try:
            puz = _recognize_once(gray, cfg_try, clf, warnings)
            break
        except RecognizeError as e:
            last_err = e
            print(f"[识别] 闭运算核={k} 失败: {e}")
    if puz is None:
        raise last_err

    # 坐标回溯映射: 处理坐标系 → 原输入图坐标系
    def to_input(x, y):
        if scale != 1.0:
            x, y = x / scale, y / scale
        if rot_ang:
            # 逆旋转: 此处括号形 [ca,-sa; sa,ca] 与 M(θ)=[c,s;-s,c] 相反,
            # 取逆须 a=+deg2rad(rot_ang)
            a = np.deg2rad(rot_ang)
            ca, sa = np.cos(a), np.sin(a)
            cx, cy = w1 / 2.0, h1 / 2.0
            dx, dy = x - cx, y - cy
            x = cx + dx * ca - dy * sa
            y = cy + dx * sa + dy * ca
        return (x + off_x, y + off_y)

    if scale != 1.0 or rot_ang or off_x or off_y:
        puz.centers = {k: to_input(x, y) for k, (x, y) in puz.centers.items()}
        puz.crop_origin = to_input(*puz.crop_origin)
    return puz


def _recognize_once(gray, cfg, clf, warnings):
    crop, wall, origin = _wall_and_crop(gray, cfg)
    ang = _skew_deg(wall)
    if ang > float(cfg["skew_max_deg"]):
        raise RecognizeError(
            f"题目倾斜约 {ang:.1f}° 超过上限 {cfg['skew_max_deg']}°, "
            f"请摆正窗口后重新框选")
    if ang > float(cfg["skew_warn_deg"]):
        warnings.append(f"题目存在 {ang:.1f}° 轻微倾斜, 已按轴对齐处理")

    comps, med_area = _interior_cells(wall, cfg)
    cell_est = float(np.sqrt(med_area))
    if cell_est < float(cfg["min_cell_px"]):
        raise RecognizeError(
            f"识别到的单元格过小({cell_est:.1f}px < {cfg['min_cell_px']}px), "
            f"请放大题目后重新框选")
    rows_y, cols_x, step_y, step_x, cell_est = _layout_from_components(
        comps, med_area, cfg)
    exist, conflict = _assign_cells(comps, rows_y, cols_x)
    a, b = len(rows_y), len(cols_x)
    if conflict:
        warnings.append(f"{conflict} 个格子内部被分割(数字可能贴线), 已合并处理")
    print(f"[识别] 版面: {a} 行 x {b} 列, 格约 {step_x:.1f}x{step_y:.1f}px")

    # 存在矩阵直通求解与作答(任意形状); 不连通的海必无解, 多为框选漏格
    if not _shape_connected(exist):
        raise RecognizeError(
            "题目存在格彼此不连通(有独立抠出的区域), 海无法连通必无解; "
            "请确认框选完整包含了整座题目的所有格子")
    params = solver.shape_params(exist)

    # 数字识别
    if clf is None:
        clf = DigitClassifier(build_templates(cfg))
    half_x = step_x / 2.0
    half_y = step_y / 2.0
    # 裁剪用整格(网格线由墙体掩码负责剔除): 有些渲染字高≈格高,
    # 留边距会把数字切掉
    margin_x = 0.0
    margin_y = 0.0
    wall_dil = cv2.dilate(
        _line_wall(wall.astype(np.uint8) * 255, cell_est),
        np.ones((3, 3), np.uint8)) > 0
    nums = {}
    centers = {}
    ch, cw = crop.shape[:2]
    for r in range(a):
        for c in range(b):
            if not exist[r][c]:
                continue
            cy = rows_y[r] + origin[1]
            cx = cols_x[c] + origin[0]
            centers[(r, c)] = (cx, cy)
            x0 = int(round(cols_x[c] - half_x + margin_x))
            x1 = int(round(cols_x[c] + half_x - margin_x))
            y0 = int(round(rows_y[r] - half_y + margin_y))
            y1 = int(round(rows_y[r] + half_y - margin_y))
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(cw, x1), min(ch, y1)
            if x1c - x0c < 4 or y1c - y0c < 4:
                continue
            cell = crop[y0c:y1c, x0c:x1c]
            wpatch = wall_dil[y0c:y1c, x0c:x1c]
            value, conf, low = _read_cell_value(
                cell, clf, cfg, f"格({r + 1},{c + 1})", wpatch)
            if value:
                nums[(r, c)] = value
    print("[识别] 网格 " + Puzzle(
        a, b, exist, nums, centers, step_x, step_y, origin,
        params, warnings).describe())
    return Puzzle(a, b, exist, nums, centers, step_x, step_y, origin,
                  params, warnings)
