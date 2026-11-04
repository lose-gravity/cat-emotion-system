# -*- coding: utf-8 -*-
"""
gen_laser_dxf.py - 生成激光切割图纸(白色亚克力3mm, 榫卯插槽结构)
产物: laser/ 目录下 每个零件一个DXF + 拼版总图
尺寸约定: 板厚3mm, 榫宽10mm, 槽宽10.3mm(容差+激光割缝), 槽深3.1mm
结构: 前后板=纯开槽; 侧板=前后边带榫; 顶底板=四边带榫; 料仓=5片插槽
"""
import ezdxf
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laser")
os.makedirs(OUT, exist_ok=True)

T = 3.0          # 板厚
TAB_W = 10.0     # 榫宽
SLOT_W = 10.3    # 槽宽(3mm板+割缝补偿)
SLOT_D = 3.1     # 槽深

# ================= 尺寸 =================
FP_W, FP_H = 210, 190        # 前后板
SD_W, SD_H = 214, 190        # 侧板(深度=220-2*3)
TB_W, TB_D = 210, 214        # 顶底板
SCREEN_W, SCREEN_H = 164, 120
WIN_W, WIN_H = 158, 114
SCREEN_Y = FP_H - 12 - SCREEN_H      # 屏区底边=58
SPK_W, SPK_H = 68, 30
SPK_Y = 8
SPK_HD = 3.5
MIC_D, MIC_X, MIC_Y = 6, 184, 14
PWR_D, PWR_X, PWR_Y = 14, 105, 30
VENT_W, VENT_H, VENT_GAP = 40, 5, 12
EAR_H, EAR_BASE = 45, 30

# 料仓(120x80x60, 前脸带出粮口30x25)
HOP_L, HOP_W, HOP_H = 120, 80, 60
CHUTE_W, CHUTE_H = 30, 25
SLOPE_ANG = 18

# 舵机座(SG90: 23x12.2x29)
SERVO_L, SERVO_W, SERVO_H = 23, 12.2, 29


def tab_positions(length, n):
    """沿一边均匀分布n个榫的位置(中心x)"""
    if n == 1:
        return [length / 2]
    step = (length - 20) / (n - 1)
    return [10 + i * step for i in range(n)]


def draw_rect(doc, x0, y0, w, h):
    msp = doc.modelspace()
    msp.add_lwpolyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h),
                        (x0, y0 + h)], close=True)


def draw_tab(doc, x, y, direction):
    """在(x,y)处画一个榫, direction: 'N'上 'S'下 'E'右 'W'左(向外凸出3mm)"""
    msp = doc.modelspace()
    if direction == 'N':
        msp.add_lwpolyline([(x - TAB_W/2, y), (x - TAB_W/2, y + T), (x + TAB_W/2, y + T),
                            (x + TAB_W/2, y)], close=True)
    elif direction == 'S':
        msp.add_lwpolyline([(x - TAB_W/2, y), (x - TAB_W/2, y - T), (x + TAB_W/2, y - T),
                            (x + TAB_W/2, y)], close=True)
    elif direction == 'E':
        msp.add_lwpolyline([(x, y - TAB_W/2), (x + T, y - TAB_W/2), (x + T, y + TAB_W/2),
                            (x, y + TAB_W/2)], close=True)
    elif direction == 'W':
        msp.add_lwpolyline([(x, y - TAB_W/2), (x - T, y - TAB_W/2), (x - T, y + TAB_W/2),
                            (x, y + TAB_W/2)], close=True)


def draw_slot(doc, x, y, inward_x, inward_y):
    """在边上(x,y)处向板内切槽(凹口): inward_x/inward_y=+1或-1表示板内方向"""
    msp = doc.modelspace()
    if inward_y != 0:   # 水平边上的槽
        y0 = min(y, y + inward_y * SLOT_D)
        msp.add_lwpolyline([(x - SLOT_W/2, y0), (x + SLOT_W/2, y0),
                            (x + SLOT_W/2, y0 + SLOT_D), (x - SLOT_W/2, y0 + SLOT_D)],
                           close=True)
    else:               # 垂直边上的槽
        x0 = min(x, x + inward_x * SLOT_D)
        msp.add_lwpolyline([(x0, y - SLOT_W/2), (x0, y + SLOT_W/2),
                            (x0 + SLOT_D, y + SLOT_W/2), (x0 + SLOT_D, y - SLOT_W/2)],
                           close=True)


def panel_with_slots(name, w, h, n_tb, n_lr):
    """面板: 四周开槽(槽位置与相邻板榫对齐)"""
    doc = ezdxf.new('R2010')
    draw_rect(doc, 0, 0, w, h)
    for x in tab_positions(w, n_tb):        # 上边槽(向内下凹)
        draw_slot(doc, x, h, 0, -1)
    for x in tab_positions(w, n_tb):        # 下边槽(向内上凹)
        draw_slot(doc, x, 0, 0, 1)
    for y in tab_positions(h, n_lr):        # 左/右边槽(向内凹)
        draw_slot(doc, 0, y, 1, 0)
        draw_slot(doc, w, y, -1, 0)
    doc.saveas(os.path.join(OUT, name + ".dxf"))
    return doc


def panel_with_tabs(name, w, h, n_tb, n_lr):
    """面板: 四边带榫"""
    doc = ezdxf.new('R2010')
    draw_rect(doc, 0, 0, w, h)
    for x in tab_positions(w, n_tb):
        draw_tab(doc, x, h, 'N')
        draw_tab(doc, x, 0, 'S')
    for y in tab_positions(h, n_lr):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, w, y, 'E')
    doc.saveas(os.path.join(OUT, name + ".dxf"))
    return doc


def front_panel():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    # 外框+四周槽(上下各6榫位, 左右各5)
    draw_rect(doc, 0, 0, FP_W, FP_H)
    for x in tab_positions(FP_W, 6):
        draw_slot(doc, x, FP_H, 0, -1)
        draw_slot(doc, x, 0, 0, 1)
    for y in tab_positions(FP_H, 5):
        draw_slot(doc, 0, y, 1, 0)
        draw_slot(doc, FP_W, y, -1, 0)
    # 屏幕可视窗
    wx = (FP_W - WIN_W) / 2
    wy = SCREEN_Y + (SCREEN_H - WIN_H) / 2
    msp.add_lwpolyline([(wx, wy), (wx + WIN_W, wy), (wx + WIN_W, wy + WIN_H),
                        (wx, wy + WIN_H)], close=True)
    # 屏幕4个安装孔(3mm)
    for hx in [2, SCREEN_W - 2]:
        for hy in [2, SCREEN_H - 2]:
            msp.add_circle(((FP_W - SCREEN_W)/2 + hx, SCREEN_Y + hy), 1.5)
    # 喇叭孔阵(3mm, 5mm间距)
    for gx in range(8, SPK_W - 4, 5):
        for gy in range(5, SPK_H - 4, 5):
            msp.add_circle(((FP_W - SPK_W)/2 + gx, SPK_Y + gy), 1.5)
    # 喇叭安装孔(3.5mm)
    for hx in [1, SPK_W - 1]:
        for hy in [1, SPK_H - 1]:
            msp.add_circle(((FP_W - SPK_W)/2 + hx, SPK_Y + hy), 1.75)
    # 麦克风孔
    msp.add_circle((MIC_X, MIC_Y), MIC_D / 2)
    doc.saveas(os.path.join(OUT, "front_panel.dxf"))


def back_panel():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, FP_W, FP_H)
    for x in tab_positions(FP_W, 6):
        draw_slot(doc, x, FP_H, 0, -1)
        draw_slot(doc, x, 0, 0, 1)
    for y in tab_positions(FP_H, 5):
        draw_slot(doc, 0, y, 1, 0)
        draw_slot(doc, FP_W, y, -1, 0)
    # 电源孔14mm
    msp.add_circle((PWR_X, PWR_Y), PWR_D / 2)
    # 散热格栅3条
    for i in range(3):
        vy = 70 + i * VENT_GAP
        msp.add_lwpolyline([(FP_W/2 - VENT_W/2, vy), (FP_W/2 + VENT_W/2, vy),
                            (FP_W/2 + VENT_W/2, vy + VENT_H),
                            (FP_W/2 - VENT_W/2, vy + VENT_H)], close=True)
    doc.saveas(os.path.join(OUT, "back_panel.dxf"))


def side_panel():
    """侧板: 前后边带榫(插前后板), 上下边开槽(接顶底板)"""
    doc = ezdxf.new('R2010')
    draw_rect(doc, 0, 0, SD_W, SD_H)
    # 前后边: 5个榫(与前后板左右边槽对齐)
    for y in tab_positions(SD_H, 5):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, SD_W, y, 'E')
    # 上下边: 槽(与顶底板短边榫对齐, 顶底板210宽对应侧板槽间距同前后板6个)
    for x in tab_positions(SD_W, 6):
        draw_slot(doc, x, SD_H, 0, -1)
        draw_slot(doc, x, 0, 0, 1)
    doc.saveas(os.path.join(OUT, "side_panel.dxf"))


def top_panel():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, TB_W, TB_D)
    for x in tab_positions(TB_W, 6):
        draw_tab(doc, x, TB_D, 'N')
        draw_tab(doc, x, 0, 'S')
    for y in tab_positions(TB_D, 6):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, TB_W, y, 'E')
    # 舵机座固定孔(前边沿, M2.5)
    for x in [TB_W/2 - 15, TB_W/2 + 15]:
        msp.add_circle((x, 6), 1.25)
    # 猫耳插槽(前两角): 31宽x3.2, 距前边20
    for cx in [30, TB_W - 30]:
        msp.add_lwpolyline([(cx - 15.5, 18.4), (cx + 15.5, 18.4),
                            (cx + 15.5, 21.6), (cx - 15.5, 21.6)], close=True)
    # 料仓底部固定槽(4条, 料仓底120x80, 居中靠前)
    hx0, hy0 = (TB_W - HOP_L) / 2, 25
    for sx, sy, rot in [(hx0 + 10, hy0 + 10, 0), (hx0 + HOP_L - 10, hy0 + 10, 0),
                        (hx0 + 10, hy0 + HOP_W - 10, 0), (hx0 + HOP_L - 10, hy0 + HOP_W - 10, 0)]:
        msp.add_lwpolyline([(sx - SLOT_W/2, sy - SLOT_W/2), (sx + SLOT_W/2, sy - SLOT_W/2),
                            (sx + SLOT_W/2, sy + SLOT_W/2), (sx - SLOT_W/2, sy + SLOT_W/2)],
                           close=True)
    doc.saveas(os.path.join(OUT, "top_panel.dxf"))


def bottom_panel():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, TB_W, TB_D)
    for x in tab_positions(TB_W, 6):
        draw_tab(doc, x, TB_D, 'N')
        draw_tab(doc, x, 0, 'S')
    for y in tab_positions(TB_D, 6):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, TB_W, y, 'E')
    doc.saveas(os.path.join(OUT, "bottom_panel.dxf"))


def hopper_parts():
    """料仓5片: 前(带出粮口)/后/左/右/底 + 斜板"""
    # 前脸: 120x60, 底部中间开出粮口30x25, 左右边带榫, 底边带榫
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, HOP_L, HOP_H)
    # 出粮口(底部中间, 从底边向上切30x25)
    msp.add_lwpolyline([(HOP_L/2 - CHUTE_W/2, 0), (HOP_L/2 + CHUTE_W/2, 0),
                        (HOP_L/2 + CHUTE_W/2, CHUTE_H),
                        (HOP_L/2 - CHUTE_W/2, CHUTE_H)], close=True)
    # 左右边榫(接左右片前边槽)
    for y in tab_positions(HOP_H, 3):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, HOP_L, y, 'E')
    # 底边榫(接底板)
    for x in tab_positions(HOP_L, 4):
        if abs(x - HOP_L/2) > CHUTE_W/2 + TAB_W:   # 避开出粮口
            draw_tab(doc, x, 0, 'S')
    doc.saveas(os.path.join(OUT, "hopper_front.dxf"))

    # 后脸: 120x60
    doc = ezdxf.new('R2010')
    draw_rect(doc, 0, 0, HOP_L, HOP_H)
    for y in tab_positions(HOP_H, 3):
        draw_tab(doc, 0, y, 'W')
        draw_tab(doc, HOP_L, y, 'E')
    for x in tab_positions(HOP_L, 4):
        draw_tab(doc, x, 0, 'S')
    doc.saveas(os.path.join(OUT, "hopper_back.dxf"))

    # 左右片: 80x60, 前/后边带榫, 底边带榫, 斜板槽(18度斜线槽)
    for name, flip in [("hopper_left", 0), ("hopper_right", 1)]:
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        draw_rect(doc, 0, 0, HOP_W, HOP_H)
        for z in tab_positions(HOP_H, 3):      # 前边槽(接前/后脸的侧边榫)
            draw_slot(doc, 0, z, 1, 0)
        for z in tab_positions(HOP_H, 3):      # 后边槽
            draw_slot(doc, HOP_W, z, -1, 0)
        for x in tab_positions(HOP_W, 3):
            draw_tab(doc, x, 0, 'S')
        # 斜板槽: 从后上(80,38)到前下(0,8)的斜线槽(板厚3.2宽)
        doc.saveas(os.path.join(OUT, name + ".dxf"))

    # 底板: 120x80, 四边榫
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, HOP_L, HOP_W)
    # 四边开槽(开口在边): 与四壁底边榫匹配
    # 后边4槽(后脸底榫4个); 前边2槽(前脸底榫只有2个, 出粮口处无榫)
    for x in tab_positions(HOP_L, 4):
        draw_slot(doc, x, HOP_W, 0, -1)
    for x in [10, HOP_L - 10]:
        draw_slot(doc, x, 0, 0, 1)
    for y in tab_positions(HOP_W, 3):
        draw_slot(doc, 0, y, 1, 0)
        draw_slot(doc, HOP_L, y, -1, 0)
    # 舵机轴孔Φ6: 出粮口中心(x=60), 距前边10mm
    msp.add_circle((HOP_L / 2, 10), 3.0)
    doc.saveas(os.path.join(OUT, "hopper_bottom.dxf"))

    # (斜板已移除: 出粮改用舵机拨轮方案))


def small_parts():
    # 猫耳
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (EAR_BASE, 0), (EAR_BASE * 0.6, EAR_H),
                        (0, EAR_H * 0.35)], close=True)
    msp.add_lwpolyline([(0, -1), (EAR_BASE, -1), (EAR_BASE, -9), (0, -9)], close=True)
    doc.saveas(os.path.join(OUT, "cat_ear.dxf"))

    # 舵机座: 底板(35x23, 2固定孔+4槽) + 两片侧壁(35x33, 2榫+2夹紧孔)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, SERVO_L + 12, SERVO_W + 10)
    for x in [(SERVO_L + 12)/2 - 15, (SERVO_L + 12)/2 + 15]:
        msp.add_circle((x, (SERVO_W + 10)/2), 1.25)
    # 侧壁槽(2条x2)
    for sx in [6, SERVO_L + 6]:
        msp.add_lwpolyline([(sx - SLOT_W/2, 4.8), (sx + SLOT_W/2, 4.8),
                            (sx + SLOT_W/2, 4.8 + SLOT_D), (sx - SLOT_W/2, 4.8 + SLOT_D)],
                           close=True)
        msp.add_lwpolyline([(sx - SLOT_W/2, SERVO_W + 5.2), (sx + SLOT_W/2, SERVO_W + 5.2),
                            (sx + SLOT_W/2, SERVO_W + 5.2 + SLOT_D),
                            (sx - SLOT_W/2, SERVO_W + 5.2 + SLOT_D)], close=True)
    doc.saveas(os.path.join(OUT, "servo_base.dxf"))

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, SERVO_L + 12, SERVO_H + 4)
    for x in [6, SERVO_L + 6]:
        draw_tab(doc, x, 0, 'S')
    # 舵机夹紧孔(M2.2, 两侧)
    msp.add_circle((SERVO_L/2 + 6 - 5, SERVO_H/2 + 2), 1.1)
    msp.add_circle((SERVO_L/2 + 6 + 5, SERVO_H/2 + 2), 1.1)
    doc.saveas(os.path.join(OUT, "servo_wall.dxf"))

    # 出粮拨片(缺口圆盘式): 圆盘36堵住出粮口, 100度扇形缺口放粮
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    # 外圆36 + 切掉100度扇形(从-40度到60度)
    msp.add_circle((0, 0), 18)
    ang_open = 100
    start = math.radians(-40)
    end = math.radians(60)
    # 扇形缺口的两条直边+圆弧(简化: 缺口画成从圆心到圆周的三角区)
    msp.add_lwpolyline([(0, 0),
                        (18 * math.cos(start), 18 * math.sin(start)),
                        (18 * math.cos(end), 18 * math.sin(end))],
                       close=True)
    msp.add_circle((0, 0), 3.0)     # 轴孔Φ6(套舵机轴/舵机臂)
    doc.saveas(os.path.join(OUT, "dispenser_vane.dxf"))

    # 元件安装板: 50x25, 功放双孔3mm
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    draw_rect(doc, 0, 0, 50, 25)
    for y in [2, 18.5]:
        msp.add_circle((15, y), 1.5)
    doc.saveas(os.path.join(OUT, "module_plate.dxf"))

    # 板子定位片 x4: 20x10
    doc = ezdxf.new('R2010')
    draw_rect(doc, 0, 0, 20, 10)
    doc.saveas(os.path.join(OUT, "board_locator.dxf"))


def main():
    front_panel()
    back_panel()
    side_panel()
    top_panel()
    bottom_panel()
    hopper_parts()
    small_parts()
    files = sorted(os.listdir(OUT))
    print(f"共 {len(files)} 个DXF:")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
