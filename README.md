# 益智谜题屏幕自动解题合集

六个独立的解谜自动化工具，流程相同：屏幕上框选题目 → 自动识别 → 求解 → 键鼠自动作答提交。配置全部写死在代码里，无命令行参数，作答过程按 ESC 可中断。

## 包含项目

以下命令均在项目根目录执行：

| 目录 | 谜题 | 入口 |
|------|------|------|
| 数独 | 标准 / 异形数独 | `python 数独/main.py` |
| 扫雷 | 扫雷（支持超出屏幕的超大棋盘，滚动拼接） | `python 扫雷/main.py` |
| 高楼 | 摩天楼（Skyscraper） | `python 高楼/main.py` |
| 数墙 | 数墙（Nurikabe） | `python 数墙/main.py` |
| 帐篷 | 帐篷与树（Tents and Trees） | `python 帐篷/main.py` |
| 马赛克 | 马赛克（Mosaic，数字=3×3邻域涂黑数） | `python 马赛克/main.py` |

## 目标网站

题目均来自同一系列的在线解谜站（中文版前缀 `cn.`，英文版为 `www.`），各项目的根站点：

| 项目 | 网站 |
|------|------|
| 数独 | https://cn.puzzle-sudoku.com/ |
| 扫雷 | https://cn.puzzle-minesweeper.com/ |
| 马赛克 | 挂在扫雷站下：https://cn.puzzle-minesweeper.com/ 左侧菜单选"马赛克" |
| 高楼 | https://cn.puzzle-skyscrapers.com/ |
| 数墙 | https://cn.puzzle-nurikabe.com/ |
| 帐篷 | https://cn.puzzle-tents.com/ |

（异形数独在 https://cn.puzzle-jigsaw-sudoku.com/ ，数独项目同样支持。）

## 使用

```bash
# 在根目录安装依赖（一次装齐六个项目）
pip install -r requirements.txt

# 在根目录直接运行任一项目
python 数墙/main.py
```

运行后屏幕出现半透明遮罩，按住左键拖动画框框住整个棋盘（可略大于棋盘，识别会自动裁剪），松开后程序自动完成识别、求解、作答并提交。

## 环境

Python 3.7+，主要在 Windows 上使用（Linux 也可运行）。 ortools 用于数墙/帐篷的 CP-SAT 求解，scipy 用于马赛克的 MILP 求解，缺失时对应项目会自动退化为纯 Python 求解器。
