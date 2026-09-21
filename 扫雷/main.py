# -*- coding: utf-8 -*-
"""main.py — 扫雷自动识别/求解/回填 主程序.

使用方法:
    python main.py

流程:
  1. 屏幕出现半透明遮罩, 两种互相独立的选题方式(按鼠标手势自动识别):
     - 拖动框选完整题目(可略大于棋盘): 信息都在框内, 单屏识别 + 框内
       直接插旗, 全程不滚动
     - 点击两个点(左点定左边界, 右点定右边界): 适用超出屏幕的超大棋盘,
       识别与回填滚动拼接; 识别起点自动锚定到棋盘加粗顶边框
     点满两点/框选松开即开始.
  2. 程序自动:
     (一) 初始化定位: 截取首屏, 检测网格, 推算列数 n
     (二) 向下滚动拼接识别: 滚轮下滚→截图→内容对齐算偏移→拼接新行,
          直到在截图内部确认完整棋盘的真实底边
     (三) 求解: 扫雷约束求解(CSP), 确定所有雷
     (四) 滚动回填: 自底向上逐屏右键标记雷
  3. 按 ESC 随时中断.

依赖: mss opencv-python numpy scipy pyautogui keyboard
"""
import time

import numpy as np

import ms_solver as ms
import ms_auto as ma

CONFIG = {
    "stop_hotkey": "esc",
    "scroll_amount": -60,        # 向下滚轮的格数(负=向下)
    "scroll_up_amount": 30,      # 向上滚轮的格数
    "scroll_settle": 0.15,       # 滚动后等待截图稳定(浏览器渲染新位置)
    "fill_recheck_settle": 0.25, # 回填定位失败时原地重截等待
    "click_interval": 0.015,     # 连续右键标记之间的间隔
    "jitter": 1.5,               # 点击抖动像素(避免点到网格线上)
    "max_no_progress": 3,        # 连续无进展判定到底
    "max_bad_screens": 3,        # 连续低质量整屏上限, 超过即安全终止
    "scroll_retry_attempts": 6,  # 单次滚动对齐失败后允许的缩小滚动重试次数
    "g0_search_radius": 4,       # 像素预测行号附近的内容校正范围
    "max_fill_scroll_steps": 50, # 单个目标行允许的最大定位滚动次数
    "fill_direction": "bottomup",  # bottomup / topdown
    "solve_time_limit": 120.0,   # 求解时间上限(唯一解题目允许充分枚举)
}

def _main_impl():
    ma.set_dpi_aware()

    print("=" * 50)
    print("  扫雷自动识别 / 求解 / 回填")
    print("  两种独立选题方式(按手势自动识别):")
    print("  - 拖动框选完整题目: 信息都在框内, 单屏识别+直接插旗, 全程不滚动")
    print("  - 点击两个点(左点=左边界, 右点=右边界): 超大棋盘用, 滚动拼接")
    print(f"  按 {CONFIG['stop_hotkey'].upper()} 随时中断")
    print("=" * 50)

    # 选题(两种模式互相独立, 下游走不同管线)
    mode, data = ma.select_region()
    if mode == "box":
        left, top, right, bottom = data
        print(f"[框选] 区域: 左={left}, 上={top}, 右={right}, 下={bottom} "
              "(完整题目单屏模式)")
    else:
        left, top, right = data
        print(f"[点选] 左边界={left}, 右边界={right}, 顶边={top} "
              "(超大棋盘滚动模式)")

    # 注册 ESC 中断
    STOP = ma.StopFlag()
    STOP.start(CONFIG["stop_hotkey"])
    STOP.reset()

    cfg = dict(CONFIG)

    if mode == "box":
        # 框选模式: 题目完整可见于框内, 截屏区域就是框本身
        bbox = (left, top, right, bottom)
    else:
        # 点击模式: 从顶边到屏幕底部(覆盖整个可用高度, 便于滚动拼接)
        scr_w, scr_h = ma.get_screen_size()
        bbox = (left, top, right, scr_h)
    if bbox[3] <= bbox[1]:
        print("[错误] 截屏区域无效")
        return

    # ---------- 识别 ----------
    if mode == "box":
        print("\n[阶段1/4] 框选单屏识别(全程不滚动)...")
        st = ma.recognize_visible_board(bbox, STOP, cfg)
        if st is None or st.board is None:
            print("[错误] 框内未识别到完整题目, 本次不求解不回填。")
            print("       请重新框选完整题目(可略大于棋盘);")
            print("       若题目超出屏幕, 请改用点击两点模式")
            return
    else:
        print("\n[阶段1/4] 初始化定位与滚动拼接识别...")
        st = ma.stitch_recognize(bbox, STOP, cfg)
        if st is None or st.board is None:
            print("[错误] 识别失败, 请检查框选区域")
            return
    # 回填使用与实际识别一致的截屏区域
    bbox = st.bbox or bbox
    n = st.n
    print(f"[识别] 完成: {n}x{n} 棋盘, 未翻开 {int((st.board == -1).sum())} 格")

    # 自洽性检查: 数字周围"未翻开格"(含已插旗格)数必须 >= 数字.
    # 旗(-2)也是未翻开格(只是已判雷), 求解器本就按旗=已判雷参与约束;
    # 只数 -1 会把"棋盘上有旗"误报成识别矛盾。
    bad_cells = []
    flag_cnt = int((st.board == ma.FLAG).sum())
    if flag_cnt:
        print(f"[提示] 棋盘上已有 {flag_cnt} 个旗(此前人工/自动标记), "
              "将按已判雷参与求解")
    for r in range(n):
        for c in range(n):
            num = st.board[r, c]
            if num < 0:
                continue
            unop = sum(1 for dr, dc in ms.NEIGHBORS
                       if 0 <= r + dr < n and 0 <= c + dc < n
                       and st.board[r + dr, c + dc] in (ms.UNOPENED, ms.FLAG))
            if unop < num:
                bad_cells.append((r, c, num, unop))
    if bad_cells:
        print(f"[错误] 识别自洽性检查发现 {len(bad_cells)} 个矛盾数字"
              f"(数字 > 周围未翻开格数(含旗), 扫雷中不可能):")
        for r, c, num, unop in bad_cells[:10]:
            print(f"  格({r},{c}) 数字{num} 周围未翻开{unop}")
        print("[安全终止] 识别结果不自洽, 不进入求解和回填")
        return

    if STOP.stopped:
        print("[中断] 已停止")
        return

    # ---------- 求解 ----------
    print("\n[阶段2/4] 求解...")
    try:
        result = ms.solve_board(st.board, verbose=True,
                                time_limit=CONFIG["solve_time_limit"])
    except ms.InconsistentBoardError as exc:
        print(f"[错误] {exc}")
        print("[安全终止] 识别结果或求解结论违反数字约束, 不回填、不提交")
        return
    mine_cells = list(zip(*np.where(result == ms.MINE)))
    print(f"[求解] 确定 {len(mine_cells)} 个雷")
    if not mine_cells:
        print("[提示] 未确定任何雷, 可能是题目已全部翻开或求解无法收敛")
        return
    unknown_cnt = int((result == ms.UNKNOWN).sum())
    if unknown_cnt > 0:
        print(f"[错误] 仍有 {unknown_cnt} 格无法判定")

    if unknown_cnt > 0:
        print("[安全终止] 唯一解题目必须 0 未知, 本次不回填、不提交")
        return

    if STOP.stopped:
        print("[中断] 已停止")
        return

    # ---------- 回填 ----------
    if mode == "box":
        print(f"\n[阶段3/4] 框内直接插旗(全程不滚动)...")
        filled = ma.fill_visible_board(st, bbox, STOP, cfg, mine_cells)
    else:
        print(f"\n[阶段3/4] 滚动回填标记雷 ({CONFIG['fill_direction']})...")
        filled = ma.fill_from_board(st, bbox, STOP, cfg, mine_cells)
    print(f"[回填] 完成, 标记 {filled} 个雷")
    if filled != len(mine_cells):
        print(f"[安全终止] 应标记 {len(mine_cells)} 个雷, 实际仅标记 {filled} 个; "
              "不提交")
        return

    # ---------- 兜底提交 ----------
    if not STOP.stopped:
        print("\n[阶段4/4] 提交...")
        time.sleep(0.3)
        # 移动鼠标到棋盘外再按回车, 避免误触格子
        ma.move_mouse(bbox[0] - 10, bbox[1] - 10)
        ma.press_enter()
        print("[完成] 已提交. 若未自动判定, 可手动按回车.")

    if STOP.stopped:
        print("\n[中断] 程序已中断")


def main():
    try:
        _main_impl()
    except Exception:
        import traceback
        print("[异常] 程序发生未处理错误:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
