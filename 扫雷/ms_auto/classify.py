# -*- coding: utf-8 -*-
"""ms_auto.classify — 格子分类: 未翻开 / 数字(含0) / 旗子, 颜色+字形判据
"""
import cv2
import numpy as np

# BGR 参考色(OpenCV 读图)
COLOR_REF = {
    1: (255, 0, 0),      # 蓝
    2: (0, 153, 0),      # 绿
    3: (0, 0, 255),      # 红
    4: (153, 0, 0),      # 深蓝
    5: (0, 0, 153),      # 暗红
    6: (153, 153, 0),    # 青
    7: (0, 0, 0),        # 黑
    8: (153, 153, 153),  # 灰
}

UNOPENED = -1
EMPTY = 0
FLAG = -2          # 识别出旗子(回填后如需再读盘用)

TEMPLATE_SIZE = 28
DIGIT_CHROMA = 40

# 未翻开结构判据参数(尺度无关; 可被 recognize 的自适应救援临时放宽)
UNOPENED_PARAMS = {"thr_delta": 10.0, "floor": 216.0, "band_div": 3,
                   "band_cap": 10, "ratio": 0.55}

# 8 的字形特征: 两个孔; 0 只有一个孔. 灰色字形需用孔数区分 0 与 8.


def _glyph_hole_count(glyph):
    """数字形内部孔数(0 有 1 个孔, 8 有 2 个孔, 7 无孔)."""
    if glyph is None or glyph.size == 0 or glyph.max() == 0:
        return 0
    inv = cv2.bitwise_not(glyph)
    ncc, lab, stats, cent = cv2.connectedComponentsWithStats(inv, 8)
    holes = 0
    for i in range(1, ncc):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= inv.shape[1] or y + h >= inv.shape[0]:
            continue
        if area >= 3:
            holes += 1
    return holes


# 单元格基本属性

def cell_is_unopened(cell_bgr, cell_gray=None):
    """未翻开格特征(全部判据尺度无关, 任何格距走同一条路径).

    输入应为"整格"切片(含贴边的网格线行/列), 各判据:

    1) 全局高光占比(>235 超过 6%)—— 大格距/整块提亮主题下未翻开格的
       经典特征; 比例判据, 与像素尺寸无关。
    2) 左上双亮边(相对判据, 主判据): 本站翻开格与未翻开格格体同为中灰,
       唯一稳定区分是未翻开格"凸起按钮"的上/左 1~2px 亮带。在贴近上线/
       左线的带窗内(宽度 = 格边长/5, 3~8px)找一条横贯≥55% 的亮行和一条
       纵贯≥55% 的亮列, 亮度按"格体中位数 + delta"相对定义——绝对阈值
       和固定像素窗口在任何缩放下的假设都不成立, 相对+比例判据才泛化。
    """
    if cell_gray is None:
        cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = cell_gray.shape
    if h < 3 or w < 3:
        return False
    bright_ratio = float((cell_gray > 235).mean())
    if bright_ratio > 0.06:
        return True

    prm = UNOPENED_PARAMS
    # 格体基线用众数而非中位数: 数字笔画(尤其大号的 0/8)在中心区可占
    # ~40%+ 面积, 中位数会被拉进笔画灰度而使阈值崩塌, 把翻开格误判成
    # 未翻开; 笔画永远不可能成为众数, 众数恒为格体色。
    cen = cell_gray[h // 4: max(h // 4 + 1, h - h // 4),
                    w // 4: max(w // 4 + 1, w - w // 4)]
    hist = np.bincount(cen.ravel())
    body = float(np.argmax(hist))
    thr = max(body + prm["thr_delta"], prm["floor"])
    # 带窗与格边长成比例(渲染相位漂移 ~4px 也落在窗内), 不设绝对尺寸门槛
    band = max(4, min(prm["band_cap"], min(h, w) // prm["band_div"]))
    row_frac = (cell_gray[:band, :] >= thr).mean(axis=1)
    col_frac = (cell_gray[:, :band] >= thr).mean(axis=0)
    return (float(row_frac.max()) >= prm["ratio"]
            and float(col_frac.max()) >= prm["ratio"])
    return False




def _ink_mask(cell_bgr, cell_gray, margin_frac=0.12):
    """格子中心区域的墨迹掩码(彩色数字或深色数字)."""
    hh, ww = cell_gray.shape
    cy0, cy1 = int(hh * margin_frac), int(hh * (1 - margin_frac))
    cx0, cx1 = int(ww * margin_frac), int(ww * (1 - margin_frac))
    if cy1 - cy0 < 4 or cx1 - cx0 < 4:
        return None
    sub = cell_bgr[cy0:cy1, cx0:cx1]
    subg = cell_gray[cy0:cy1, cx0:cx1]
    chroma = (np.abs(sub[:, :, 0].astype(int) - sub[:, :, 1]) +
              np.abs(sub[:, :, 1] - sub[:, :, 2]) +
              np.abs(sub[:, :, 2] - sub[:, :, 0]))
    ink = (chroma > DIGIT_CHROMA) | ((subg < 170) & (chroma < DIGIT_CHROMA))
    if ink.sum() < 8:
        return None
    return ink


def classify_by_color(cell_bgr, cell_gray, chroma_floor=10):
    """颜色主分类: 返回 (数字, 中位墨色). 无墨迹时返回 (None, None).

    数字范围: 0-8. 0 表示"周围无雷"的数字, 与空白(无墨迹)不同.
    """
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is None:
        return None, None
    hh, ww = cell_gray.shape
    cy0, cy1 = int(hh * 0.12), int(hh * 0.88)
    cx0, cx1 = int(ww * 0.12), int(ww * 0.88)
    px = cell_bgr[cy0:cy1, cx0:cx1][ink]
    chp = (np.abs(px[:, 0].astype(int) - px[:, 1]) +
           np.abs(px[:, 1] - px[:, 2]) +
           np.abs(px[:, 2] - px[:, 0]))
    chrom_px = px[chp > DIGIT_CHROMA]
    if len(chrom_px) > chroma_floor:
        med = np.median(chrom_px, axis=0)
        best, bd = 0, 1e9
        for d, ref in COLOR_REF.items():
            dist = np.abs(med - np.array(ref)).sum()
            if dist < bd:
                bd, best = dist, d
        return best, med
    # 无彩色 -> 灰/黑数字: 灰=0(单环)或8(双环), 黑=7
    medg = np.median(px, axis=0)
    if medg.mean() > 70:
        # 灰色: 用孔数区分 0 与 8
        glyph = extract_glyph(cell_bgr, cell_gray)
        holes = _glyph_hole_count(glyph)
        return (8 if holes >= 2 else 0), medg
    return 7, medg


def extract_glyph(cell_bgr, cell_gray):
    """提取归一化 28x28 二值字形."""
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is None:
        return None
    g = (ink.astype(np.uint8)) * 255
    ys, xs = np.where(g)
    if len(ys) == 0:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = (TEMPLATE_SIZE - 6) / max(g.shape)
    nw = max(1, int(g.shape[1] * sc))
    nh = max(1, int(g.shape[0] * sc))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    y0c = (TEMPLATE_SIZE - nh) // 2
    x0c = (TEMPLATE_SIZE - nw) // 2
    canvas[y0c:y0c + nh, x0c:x0c + nw] = g
    return canvas


def classify_cell(cell_bgr):
    """综合分类一个格子: 返回 (值, 详情dict).

    值: -1 未翻开, 0 空白, 1-8 数字, -2 旗.
    注意: 数字 0 也是"数字"(周围无雷), 与"空白"不同.
    分类只依赖颜色/形状; 模板匹配对最终值无影响, 已移除(见 classify_screen).
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return EMPTY, {}
    cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 未翻开(高亮浮雕) -> 旗或未翻开
    if cell_is_unopened(cell_bgr, cell_gray):
        if _flag_like(cell_bgr):
            return FLAG, {"type": "flag"}
        return UNOPENED, {"type": "unopened"}

    # 2. 数字墨迹(彩色/深色)
    ink = _ink_mask(cell_bgr, cell_gray)
    if ink is not None:
        v_color, med = classify_by_color(cell_bgr, cell_gray)
        if v_color is not None:   # 有墨迹 => 是数字(含0), 不是空白
            return v_color, {"type": "digit", "color": v_color, "med": med}

    return EMPTY, {"type": "empty"}


def _flag_like(cell_bgr):
    """旗子特征: 红色三角旗面在格子上半部, 红色像素适中.

    与红色数字(3/5)区分: 数字居中(ycenter~0.47)且红色像素多(>80);
    旗子偏上(ycenter~0.35)且红色像素少(40-46).
    """
    b = cell_bgr[:, :, 0].astype(int)
    g = cell_bgr[:, :, 1].astype(int)
    r = cell_bgr[:, :, 2].astype(int)
    red = (r > 130) & (r > g + 60) & (r > b + 60)
    n_red = int(red.sum())
    if not (15 <= n_red <= 70):
        return False
    h, w = red.shape
    ys, xs = np.where(red)
    if len(ys) == 0:
        return False
    y_center = ys.mean() / h
    return y_center < 0.42


def classify_screen(img_bgr, grid, templates=None):
    """按规整网格把一屏切成格子并逐个分类.

    templates 参数保留仅为接口兼容(历史用途是模板匹配诊断); 分类结果完全
    由颜色/形状决定, 与模板无关。逐格对模板做 matchTemplate 会让大棋盘识别
    慢约 7 倍, 故不再使用。

    返回:
      values  m 行 x n 列 的整型数组(-1 未翻开 / 0 空白 / 1-8 数字)
      details 对应细节列表
      cells   BGR 格子列表(按行优先)
    """
    n = grid["n"]
    m = grid["m"]
    vx, hy = grid["vx"], grid["hy"]
    # 整格切片(线行并入): 判据全部尺度无关; 数字墨迹由 _ink_mask
    # 的 12% 边距自行避开贴线 AA。
    values = np.zeros((m, n), dtype=int)
    details = [[None] * n for _ in range(m)]
    cells = []
    for r in range(m):
        y0, y1 = hy[r], hy[r + 1]
        if y1 - y0 < 2:
            values[r, :] = UNOPENED
            continue
        for c in range(n):
            x0, x1 = vx[c], vx[c + 1]
            if x1 - x0 < 2:
                values[r, c] = UNOPENED
                continue
            cell = img_bgr[y0:y1, x0:x1]
            if cell.size == 0:
                values[r, c] = UNOPENED
                continue
            v, det = classify_cell(cell)
            values[r, c] = v
            details[r][c] = det
            cells.append(cell)
    return values, details, cells

