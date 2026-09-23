# -*- coding: utf-8 -*-
"""ms_auto — 扫雷自动化包(桌面 / 网格 / 分类 / 行匹配 / 识别 / 回填).

由原单文件 ms_auto.py(2800 行)按职责拆分而来, 对外接口保持不变:
main.py 仍以 `import ms_auto as ma` 使用本包导出的全部名称。

模块划分:
  desktop    中断热键 / 选题(框选|两点) / 截屏 / 鼠标键盘
  grid       隔线检测 / 网格规整 / 逐屏 analyze_screen / 滚动偏移 / 顶边锚定
  classify   单格分类(未翻开 / 数字含0 / 旗), 颜色+字形判据
  rows       StitchedBoard / 行指纹与行证据 / 各场景绝对行号选择
  recognize  识别管线: stitch_recognize(滚动拼接) 与 recognize_visible_board(单屏)
  fill       回填管线: fill_from_board(滚动) 与 fill_visible_board(框内直点)

关键原则(不变): 滚动偏移由图像对齐计算, 绝不依赖滚轮格数; 识别必须确认
棋盘完整(真实底边或框=完整题目)后才允许求解; 回填点击前逐行校验。
"""
from .desktop import (
    IS_WINDOWS, StopFlag, set_dpi_aware, select_region, grab_screen,
    get_screen_size, move_mouse, scroll_wheel, right_click, press_enter,
    MOVE_THRESHOLD, MIN_BOX_SIZE,
)
from .grid import (
    DARK_GRAY, SEP_THRESH, MIN_LINES,
    detect_h_bands, detect_v_bands, compute_pitch, analyze_screen,
    regularize_lines, count_grid_intervals, match_scroll_delta,
    align_screens, compute_scroll_offset, _screens_identical,
    refine_horizontal_bbox, _anchor_board_top, _left_border_top_y,
)
from .classify import (
    COLOR_REF, UNOPENED, EMPTY, FLAG, TEMPLATE_SIZE,
    cell_is_unopened, classify_by_color, extract_glyph, classify_cell,
    classify_screen,
)
from .rows import (
    StitchedBoard, _normalize_row, _row_signature, _row_values_at,
    _fill_row_evidence, _verify_row_against_board, _use_reference_columns,
    _choose_screen_g0, _screen_match_is_reliable, _choose_fill_g0,
    _fill_match_is_reliable, _choose_bottomup_fill_window,
    _fuse_row_observations, _row_similarity,
)
from .recognize import (
    stitch_recognize, recognize_visible_board, _locate_board,
    _locate_board_static,
)
from .fill import fill_from_board, fill_visible_board

__all__ = [
    "IS_WINDOWS", "StopFlag", "set_dpi_aware", "select_region",
    "grab_screen", "get_screen_size", "move_mouse", "scroll_wheel",
    "right_click", "press_enter", "MOVE_THRESHOLD", "MIN_BOX_SIZE",
    "DARK_GRAY", "SEP_THRESH", "MIN_LINES", "detect_h_bands",
    "detect_v_bands", "compute_pitch", "analyze_screen", "regularize_lines",
    "count_grid_intervals", "match_scroll_delta", "align_screens",
    "compute_scroll_offset", "refine_horizontal_bbox",
    "COLOR_REF", "UNOPENED", "EMPTY", "FLAG", "TEMPLATE_SIZE",
    "cell_is_unopened", "classify_by_color", "extract_glyph",
    "classify_cell", "classify_screen", "StitchedBoard",
    "stitch_recognize", "recognize_visible_board",
    "fill_from_board", "fill_visible_board",
]
