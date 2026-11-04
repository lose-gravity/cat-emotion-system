# -*- coding: utf-8 -*-
"""
gen_dwg.py v2 - 工程制图版: 图框+标题栏+全套尺寸标注
标注内容: 外形尺寸/孔位尺寸/孔径/榫槽尺寸/位置尺寸, 全部按工程制图规范(箭头+尺寸线+引出线)
"""
import os
import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle as MCircle, Rectangle
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
LASER = os.path.join(BASE, "laser")
OUT = os.path.join(BASE, "工程图")
os.makedirs(OUT, exist_ok=True)
DPI = 300
SHEET_W, SHEET_H = 420, 297
FRAME = 10
TITLE_H = 28

PARTS = {
    'front_panel':     ('前面板', 1),
    'back_panel':      ('背板', 1),
    'side_panel':      ('侧板', 2),
    'bottom_panel':    ('底板', 1),
    'top_panel':       ('顶板', 1),
    'hopper_front':    ('料仓前脸', 1),
    'hopper_back':     ('料仓后脸', 1),
    'hopper_left':     ('料仓左片', 1),
    'hopper_right':    ('料仓右片', 1),
    'hopper_bottom':   ('料仓底板', 1),
    'cat_ear':         ('猫耳', 2),
    'servo_base':      ('舵机座底板', 1),
    'servo_wall':      ('舵机座侧壁', 2),
    'dispenser_vane':  ('出粮拨片', 1),
    'module_plate':    ('元件安装板', 1),
    'board_locator':   ('板子定位片', 4),
}


def get_entities(path):
    doc = ezdxf.readfile(path)
    polys, circles = [], []
    for e in doc.modelspace():
        if e.dxftype() == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                polys.append(pts)
        elif e.dxftype() == 'CIRCLE':
            c = e.dxf.center
            circles.append((c.x, c.y, e.dxf.radius))
    return polys, circles


def bounds(polys, circles):
    xs, ys = [], []
    for p in polys:
        for x, y in p:
            xs.append(x); ys.append(y)
    for x, y, r in circles:
        xs += [x - r, x + r]; ys += [y - r, y + r]
    return min(xs), max(xs), min(ys), max(ys)


class Dim:
    def __init__(self, ax, ox, oy, s, x0, y0):
        self.ax = ax; self.ox = ox; self.oy = oy; self.s = s
        self.x0 = x0; self.y0 = y0

    def P(self, x, y):
        return (self.ox + (x - self.x0) * self.s, self.oy + (y - self.y0) * self.s)

    def h(self, xa, xb, y, text, off=-10):
        p1 = self.P(xa, y); p2 = self.P(xb, y)
        pl1 = (p1[0], p1[1] + off); pl2 = (p2[0], p2[1] + off)
        self.ax.annotate('', xy=(p1[0], p1[1]), xytext=(pl1[0], pl1[1]),
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='#333'))
        self.ax.annotate('', xy=(p2[0], p2[1]), xytext=(pl2[0], pl2[1]),
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='#333'))
        self.ax.annotate('', xy=(pl2[0], pl2[1]), xytext=(pl1[0], pl1[1]),
                         arrowprops=dict(arrowstyle='<->', lw=0.8, color='#333'))
        self.ax.text((pl1[0] + pl2[0]) / 2, (pl1[1] + pl2[1]) / 2, text,
                     ha='center', va='bottom', fontsize=7)

    def v(self, x, ya, yb, text, off=-10):
        p1 = self.P(x, ya); p2 = self.P(x, yb)
        pl1 = (p1[0] + off, p1[1]); pl2 = (p2[0] + off, p2[1])
        self.ax.annotate('', xy=(p1[0], p1[1]), xytext=(pl1[0], pl1[1]),
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='#333'))
        self.ax.annotate('', xy=(p2[0], p2[1]), xytext=(pl2[0], pl2[1]),
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='#333'))
        self.ax.annotate('', xy=(pl2[0], pl2[1]), xytext=(pl1[0], pl1[1]),
                         arrowprops=dict(arrowstyle='<->', lw=0.8, color='#333'))
        self.ax.text((pl1[0] + pl2[0]) / 2 - 4, (pl1[1] + pl2[1]) / 2, text,
                     ha='right', va='center', fontsize=7, rotation=90)

    def dia(self, x, y, d, text):
        p = self.P(x, y)
        r = d / 2 * self.s + 8
        ex = p[0] + r * 0.7071; ey = p[1] + r * 0.7071
        self.ax.annotate('', xy=(ex, ey),
                         xytext=(p[0] + d / 2 * self.s * 0.7071, p[1] + d / 2 * self.s * 0.7071),
                         arrowprops=dict(arrowstyle='->', lw=0.5, color='#333'))
        self.ax.text(ex, ey, text, fontsize=7, va='bottom', ha='left')

    def note(self, x, y, text, dx=8, dy=8):
        p = self.P(x, y)
        self.ax.annotate('', xy=p, xytext=(p[0] + dx, p[1] + dy),
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='#555'))
        self.ax.text(p[0] + dx, p[1] + dy, text, fontsize=7, va='bottom', ha='left')


def dims_for(name, D):
    if name == 'front_panel':
        D.h(0, 210, 0, '210', off=-14)
        D.v(210, 0, 190, '190', off=14)
        wx = (210 - 158) / 2
        D.h(wx, wx + 158, 175, '158')
        D.v(wx, 61, 175, '114')
        D.v(wx, 175, 190, '15', off=28)
        D.note(26 + 2, 58 + 2, '4xΦ3 屏安装孔 距屏边2', 10, 4)
        D.h(71, 139, 8, '68', off=-24)
        D.v(71, 8, 38, '30', off=8)
        D.note(74, 24, '喇叭孔阵 Φ3 间距5', 14, -10)
        D.note(72, 9, '4xΦ3.5 喇叭安装孔', 12, -26)
        D.dia(184, 14, 6, 'Φ6 麦克风孔')
        D.note(10, 0, '槽 10.3x3.1 (六边共22个)', 18, -2)
        return
    if name == 'back_panel':
        D.h(0, 210, 0, '210', off=-14)
        D.v(210, 0, 190, '190', off=14)
        D.dia(105, 30, 14, 'Φ14 电源孔')
        D.note(85, 70, '散热格栅 40x5 x3 间距12', 14, 22)
        D.note(10, 0, '槽 10.3x3.1', 18, -2)
        return
    if name == 'side_panel':
        D.h(0, 214, 0, '214', off=-14)
        D.v(214, 0, 190, '190', off=14)
        D.note(0, 10, '榫 10x3 x5 (前后边)', 16, 4)
        D.note(10, 0, '槽 10.3x3.1 x6 (上下边)', 18, -6)
        return
    if name == 'bottom_panel':
        D.h(0, 210, 0, '210', off=-14)
        D.v(210, 0, 214, '214', off=14)
        D.note(10, 0, '榫 10x3 x6 (四边共24个)', 18, -4)
        return
    if name == 'top_panel':
        D.h(0, 210, 0, '210', off=-14)
        D.v(210, 0, 214, '214', off=14)
        D.note(90, 6, '2xΦ2.5 舵机座固定孔 距前6', 14, 28)
        D.note(14.5, 18.4, '猫耳槽 31x3.2 x2', 14, 20)
        D.note(55, 25, '料仓槽 10.3x10.3 x4', 14, 30)
        return
    if name == 'hopper_front':
        D.h(0, 120, 0, '120', off=-14)
        D.v(120, 0, 60, '60', off=14)
        D.h(45, 75, 0, '30', off=-24)
        D.v(45, 0, 25, '25', off=8)
        D.note(45, 12, '出粮口', 14, -16)
        D.note(0, 30, '榫 10x3 x3 (左右边)', 16, 4)
        return
    if name == 'hopper_back':
        D.h(0, 120, 0, '120', off=-14)
        D.v(120, 0, 60, '60', off=14)
        D.note(0, 30, '榫 10x3 x3 (左右边)', 16, 4)
        return
    if name in ('hopper_left', 'hopper_right'):
        D.h(0, 80, 0, '80', off=-14)
        D.v(80, 0, 60, '60', off=14)
        D.note(40, 25, '斜板槽 (18度)', 14, 10)
        return
    if name == 'hopper_bottom':
        D.h(0, 120, 0, '120', off=-14)
        D.v(120, 0, 80, '80', off=14)
        D.note(10, 0, '槽 (四边, 接四壁榫)', 18, -4)
        return
    if name == 'hopper_slope':
        D.h(0, 112, 0, '112', off=-14)
        D.v(0, 0, 76, '76', off=14)
        D.note(30, 20, '斜板 (插左右片斜槽)', 16, 12)
        return
    if name == 'cat_ear':
        D.h(0, 30, -1, '30', off=-14)
        D.v(30, -9, 45, '54', off=14)
        D.note(30, 20, '插条 30x8', 16, 10)
        return
    if name == 'servo_base':
        D.h(0, 35, 0, '35', off=-14)
        D.v(35, 0, 22.2, '22.2', off=14)
        D.note(2.5, 11.1, '2xΦ2.5 固定孔', 14, 8)
        D.note(6, 4.8, '侧壁槽 10.3x3.1 x4', 14, 18)
        return
    if name == 'servo_wall':
        D.h(0, 35, 0, '35', off=-14)
        D.v(35, -3, 33, '36', off=14)
        D.note(11.5, 18.5, '2xΦ2.2 舵机夹紧孔', 14, 12)
        return
    if name == 'dispenser_vane':
        D.dia(0, 0, 5, 'Φ5 轴孔')
        D.note(0, 0, '叶片 18x4 x3 毂Φ12', 16, -16)
        return
    if name == 'module_plate':
        D.h(0, 50, 0, '50', off=-14)
        D.v(50, 0, 25, '25', off=14)
        D.note(15, 2, '2xΦ3 功放安装孔', 14, 10)
        return
    if name == 'board_locator':
        D.h(0, 20, 0, '20', off=-14)
        D.v(20, 0, 10, '10', off=14)
        return


def draw_sheet(name, cn, qty, polys, circles, x0, x1, y0, y1):
    fig = plt.figure(figsize=(SHEET_W / 25.4, SHEET_H / 25.4), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SHEET_W); ax.set_ylim(0, SHEET_H)
    ax.set_aspect('equal'); ax.axis('off')

    ax.add_patch(Rectangle((FRAME, FRAME), SHEET_W - 2 * FRAME, SHEET_H - 2 * FRAME,
                           fill=False, edgecolor='black', lw=1.0))
    tb_x0, tb_w = SHEET_W - FRAME - 180, 180
    ax.add_patch(Rectangle((tb_x0, FRAME), tb_w, TITLE_H, fill=False, edgecolor='black', lw=0.8))
    widths = [45, 45, 30, 30, 30]
    cx = tb_x0
    for w in widths:
        cx += w
        ax.plot([cx, cx], [FRAME, FRAME + TITLE_H], color='black', lw=0.5)
    ax.plot([tb_x0, tb_x0 + 180], [FRAME + TITLE_H / 2, FRAME + TITLE_H / 2], color='black', lw=0.5)
    cols = [(0, 45, cn), (45, 45, '白色亚克力 3mm'), (90, 30, f'{qty}件'),
            (120, 30, '1:1'), (150, 30, 'mm')]
    for x, w, txt in cols:
        ax.text(tb_x0 + x + w / 2, FRAME + TITLE_H / 2 + 7, txt, ha='center', va='center', fontsize=8)

    part_w = x1 - x0; part_h = y1 - y0
    area_w = SHEET_W - 2 * FRAME - 70
    area_h = SHEET_H - 2 * FRAME - TITLE_H - 45
    s = min(area_w / part_w, area_h / part_h, 1.0)
    px0 = FRAME + (SHEET_W - 2 * FRAME - part_w * s) / 2
    py0 = FRAME + TITLE_H + 25

    def pt(x, y):
        return (px0 + (x - x0) * s, py0 + (y - y0) * s)

    for p in polys:
        ax.add_patch(Polygon([pt(x, y) for x, y in p], closed=True,
                             fill=False, edgecolor='black', lw=0.9))
    for x, y, r in circles:
        cx2, cy2 = pt(x, y)
        ax.add_patch(MCircle((cx2, cy2), r * s, fill=False, edgecolor='black', lw=0.9))

    D = Dim(ax, px0, py0, s, x0, y0)
    dims_for(name, D)
    if s < 1.0:
        ax.text(FRAME + 5, SHEET_H - FRAME - 8, f'图面比例 1:{1/s:.1f}', fontsize=7)

    for ext in ['jpg', 'pdf']:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=DPI,
                    facecolor='white', pad_inches=0)
    plt.close(fig)


def main():
    for name, (cn, qty) in PARTS.items():
        polys, circles = get_entities(os.path.join(LASER, name + '.dxf'))
        x0, x1, y0, y1 = bounds(polys, circles)
        draw_sheet(name, cn, qty, polys, circles, x0, x1, y0, y1)
    print(f'工程图(全套尺寸标注)完成: {len(PARTS)} 张 -> {OUT}')


if __name__ == '__main__':
    main()
