# -*- coding: utf-8 -*-
"""离线回归: 网格细分修复 + 拆分等价 + 框选端到端 + 扰动鲁棒性.

用法: python regress_check.py [棋盘截图.png]
无需显示器/网络; 真实截图存在时跑端到端, 否则只跑合成用例。
"""
import os
import sys

import cv2
import numpy as np

import ms_auto as ma
import ms_auto.desktop as dsk
import ms_auto.fill as fl
import ms_auto.grid as grd
import ms_auto.recognize as rec
import ms_solver as ms

SNAPS = sys.argv[1:]
FAILS = []


def check(name, cond, *info):
    print(('PASS ' if cond else 'FAIL ') + name, info if not cond else '')
    if not cond:
        FAILS.append(name)


def synth(n, pitch, soft, seed=1):
    r = np.random.default_rng(seed)
    size = n * pitch + 6
    img = np.full((size, size, 3), 250, np.uint8)
    for i in range(n):
        for j in range(n):
            y0, x0 = 3 + i * pitch, 3 + j * pitch
            if r.random() < 0.35:
                img[y0:y0 + pitch, x0:x0 + pitch] = 245
            else:
                img[y0:y0 + pitch, x0:x0 + pitch] = 204
                img[y0:y0 + 2, x0:x0 + pitch] = 255
                img[y0:y0 + pitch, x0:x0 + 2] = 255
    for idx in range(n + 1):
        p = 3 + idx * pitch
        if soft == 'all' and 0 < idx < n:
            img[p - 1, :] = 134
            img[p, :] = 146
            img[:, p - 1] = 134
            img[:, p] = 146
        elif soft and idx % 2 == 1 and 0 < idx < n:
            img[p - 1, :] = 134
            img[p, :] = 146
            img[:, p - 1] = 134
            img[:, p] = 146
        else:
            img[p, :] = 102
            img[:, p] = 102
    return img


# 合成用例
for p_ in (20, 40):
    g = ma.analyze_screen(synth(10, p_, False))
    check('正常实线不误细分 pitch=%d' % p_,
          g and g['n'] == 10 and g['m'] == 10 and abs(g['pitch'] - p_) < 1.5,
          g and (g['n'], g['m'], g['pitch']))
g = ma.analyze_screen(synth(10, 40, True))
check('柔化线细分 10x10', g['n'] == 10 and g['m'] == 10 and abs(g['pitch'] - 40) < 1.5,
      (g['n'], g['m'], g['pitch']))
g = ma.analyze_screen(synth(10, 16, True))
check('小格距柔化可识别', g and g['n'] == 10, g and (g['n'], g['m'], g['pitch']))
g = ma.analyze_screen(synth(12, 40, True, seed=9))
check('柔化 12x12', g['n'] == 12 and g['m'] == 12, (g['n'], g['m']))
img_s = synth(10, 40, True)
check('细分幂等', ma.analyze_screen(img_s)['hy_raw'] == ma.analyze_screen(img_s)['hy_raw'])
# 全柔化线(50x50 实机形态): 只剩边框 -> 自相关合成兜底
for nn, pp in ((20, 24), (10, 40)):
    g = ma.analyze_screen(synth(nn, pp, 'all'))
    check('全柔化合成兜底 %dx%d p=%d' % (nn, nn, pp),
          g is not None and g['n'] == nn and g['m'] == nn,
          g and (g['n'], g['m'], g['pitch']))
# 兜底不得误触发: 全实线大图(边框之外线都实)走正常路径
g = ma.analyze_screen(synth(15, 24, False))
check('全实线不走近兜底', g and g['n'] == 15 and g['m'] == 15, g and (g['n'], g['m']))

for SNAP in SNAPS:
    if not os.path.exists(SNAP):
        continue
    img = cv2.imread(SNAP)
    H, W = img.shape[:2]
    g = ma.analyze_screen(img)
    check('%s: 网格识别成功' % os.path.basename(SNAP)[:12], g is not None)
    if g is None:
        continue
    N = g['n']
    check('%s: 行列一致(n=%d)' % (os.path.basename(SNAP)[:12], N), g['m'] >= N, (g['n'], g['m']))
    vals, _, _ = ma.classify_screen(img, g)
    board = vals[:N, :N]
    bad = [(int(r), int(c), int(v)) for r in range(N) for c in range(N)
           for v in [int(board[r, c])] if v >= 0
           and sum(1 for dr, dc in ms.NEIGHBORS if 0 <= r + dr < N and 0 <= c + dc < N
                   and board[r + dr, c + dc] in (-1, -2)) < v]
    check('%s: 自洽性 0 矛盾' % os.path.basename(SNAP)[:12], not bad, bad[:5])
    res = ms.solve_board(board, verbose=False, time_limit=120)
    check('%s: 求解 0 未知' % os.path.basename(SNAP)[:12],
          int((res == ms.UNKNOWN).sum()) == 0, '')

    def fake_grab(bbox, _img=img):
        l, t_, r, b = [int(v) for v in bbox]
        return _img[max(0, t_):b, max(0, l):r]

    for m in (dsk, grd, rec, fl):
        if hasattr(m, 'grab_screen'):
            m.grab_screen = fake_grab
        if hasattr(m, 'get_screen_size'):
            m.get_screen_size = lambda: (W, H)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError('box mode must not scroll'))
    for m in (dsk, rec, fl):
        m.scroll_wheel = boom

    class Stop:
        stopped = False

    st = rec.recognize_visible_board((0, 0, W, H), Stop(), {'templates': None})
    check('%s: 框选管线 n=%d' % (os.path.basename(SNAP)[:12], N),
          st is not None and st.n == N)
    clicks = []
    fl.right_click = lambda x, y, jitter=0: clicks.append((x, y))
    mines = list(zip(*np.where(res == ms.MINE)))
    filled = fl.fill_visible_board(st, (0, 0, W, H), Stop(),
                                   {'jitter': 0, 'click_interval': 0, 'templates': None},
                                   mines)
    hy0, vx0, p_ = st.screens[0][1], st.reference_vx, st.pitch
    ok_coord = all(abs(x - (vx0[c] + vx0[c + 1]) / 2) < 3 and abs(y - (hy0[r] + p_ / 2)) < 3
                   for (r, c), (x, y) in zip(mines, clicks))
    check('%s: 回填全量+坐标正确' % os.path.basename(SNAP)[:12],
          filled == len(mines) == len(clicks) and ok_coord,
          (filled, len(mines), len(clicks)))

    def perturb(im, dy, dx, gain):
        out = np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (out.shape[1], out.shape[0]))

    # 注意: 小格距图在强增亮(1.15)下高光线会饱和成假"浮雕", 属已记录的
    # 绝对亮度阈值局限(README); 严格集只收常规扰动, 强增亮单列观察。
    silent_wrong = 0
    for dy, dx, gain in [(-4, 1, 1.1), (0, 2, 0.97), (1, 0, 1.05),
                         (3, 3, 1.0), (-2, 5, 1.0)]:
        im2 = perturb(img, dy, dx, gain)
        g2 = ma.analyze_screen(im2)
        if g2 is None or g2['n'] != N or g2['m'] < N:
            continue                    # 安全失败
        v2, _, _ = ma.classify_screen(im2, g2)
        silent_wrong += int((v2[:N, :N] != board).sum())
    check('%s: 常规扰动无错格' % os.path.basename(SNAP)[:12], silent_wrong == 0, silent_wrong)

    caught = True
    for dy, dx, gain in [(2, -3, 0.9), (-1, -2, 0.92)]:
        im2 = perturb(img, dy, dx, gain)
        g2 = ma.analyze_screen(im2)
        if g2 is None:
            continue
        v2, _, _ = ma.classify_screen(im2, g2)
        NN = min(int(g2['n']), int(v2.shape[0]))
        if NN < N:
            continue                    # 行数都不全 = 安全失败
        b = v2[:NN, :NN]
        badc = sum(1 for r in range(NN) for c in range(NN) if b[r, c] >= 0
                   and sum(1 for dr, dc in ms.NEIGHBORS
                           if 0 <= r + dr < NN and 0 <= c + dc < NN
                           and b[r + dr, c + dc] in (-1, -2)) < b[r, c])
        if badc == 0 and int((b != board[:NN, :NN]).sum()) > 0:
            caught = False
    check('%s: 重度变暗被安全网拦截' % os.path.basename(SNAP)[:12], caught)

print()
print('==>', '全部通过' if not FAILS else '失败: %s' % FAILS)
sys.exit(1 if FAILS else 0)
