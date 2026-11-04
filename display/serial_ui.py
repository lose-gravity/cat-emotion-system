# -*- coding: utf-8 -*-
"""
serial_ui.py - 猫叫情境识别系统 演示界面(PC端)
数据源: PuTTY 的会话日志文件(板子串口输出实时写入)
用法:
  1. PuTTY 里: Logging -> "All session output" -> 日志文件填 D:\vivado_ws\serial.log
  2. 板子上跑 02_m2_live.py(或02_m2_full.py)
  3. 电脑上: py -3.12 D:\vivado_ws\serial_ui.py
界面: 全屏大字显示 情绪/概率条/动作/确认进度, 可拖到扩展屏上
"""
import re
import os
import time
import tkinter as tk
from tkinter import font as tkfont

LOG = r"D:\vivado_ws\serial.log"
EMO_CN = {"affection": "梳理放松(亲昵)", "food": "等待喂食(饥饿)", "isolation": "环境焦虑(焦虑)"}
EMO_COLOR = {"affection": "#4ade80", "food": "#facc15", "isolation": "#f87171"}
ACTION_CN = {"不动作": "保持静默", "出粮": "投放猫粮(舵机执行)", "安抚": "播放安抚语音"}

root = tk.Tk()
root.title("猫叫情境识别与智能照料系统")
root.configure(bg="#0f172a")

# 自动定位: 有第二块屏就把窗口开在第二块屏上
import ctypes
from ctypes import wintypes

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

monitors = []
MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.POINTER(RECT), ctypes.c_void_p)
def _cb(h, dc, rect, data):
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    ctypes.windll.user32.GetMonitorInfoW(h, ctypes.byref(mi))
    r = mi.rcMonitor
    monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top,
                     bool(mi.dwFlags & 1)))  # dwFlags&1 = 主屏
    return 1

ctypes.windll.user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(_cb), 0)
target = next((m for m in monitors if not m[4]), None) or (monitors[0] if monitors else None)
if target and len(monitors) > 1:
    x, y, mw, mh = target[:4]
    root.overrideredirect(True)   # 无边框真全屏, 盖住任务栏
    root.geometry(f"{mw}x{mh}+{x}+{y}")
    W, H = mw, mh
else:
    root.overrideredirect(True)
    W, H = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+0+0")
big = tkfont.Font(family="Microsoft YaHei", size=44, weight="bold")
title_f = tkfont.Font(family="Microsoft YaHei", size=26, weight="bold")
mid = tkfont.Font(family="Microsoft YaHei", size=20)
small = tkfont.Font(family="Microsoft YaHei", size=16)

tk.Label(root, text="猫叫情境识别与智能照料系统", font=title_f,
         bg="#0f172a", fg="#e2e8f0").pack(pady=(40, 10))
tk.Label(root, text="基于 FPGA 硬件加速 | PYNQ-Z1 (Zynq-7020)", font=small,
         bg="#0f172a", fg="#64748b").pack()

time_lbl = tk.Label(root, text="--:--:--", font=mid, bg="#0f172a", fg="#94a3b8")
time_lbl.pack(pady=(30, 10))

emo_lbl = tk.Label(root, text="等待识别...", font=big, bg="#0f172a", fg="#e2e8f0")
emo_lbl.pack(pady=10)

conf_lbl = tk.Label(root, text="", font=mid, bg="#0f172a", fg="#cbd5e1")
conf_lbl.pack()

# 三条概率条
bars = {}
for i, (k, name) in enumerate([("affection", "亲昵/梳理放松"), ("food", "饥饿/等待喂食"), ("isolation", "焦虑/环境焦虑")]):
    row = tk.Frame(root, bg="#0f172a")
    row.pack(pady=6, padx=80, fill="x")
    tk.Label(row, text=name, font=small, width=14, anchor="w", bg="#0f172a", fg="#94a3b8").pack(side="left")
    cv = tk.Canvas(row, width=int(W * 0.5), height=22, bg="#1e293b", highlightthickness=0)
    cv.pack(side="left", padx=8)
    pct = tk.Label(row, text="0%", font=small, width=5, anchor="e", bg="#0f172a", fg="#e2e8f0")
    pct.pack(side="left")
    bars[k] = (cv, pct)

tk.Frame(root, height=30, bg="#0f172a").pack()
act_lbl = tk.Label(root, text="执行动作: --", font=tkfont.Font(family="Microsoft YaHei", size=30, weight="bold"),
                   bg="#0f172a", fg="#38bdf8")
act_lbl.pack(pady=10)
soothe_lbl = tk.Label(root, text="", font=mid, bg="#0f172a", fg="#fbbf24")
soothe_lbl.pack()
ip_lbl = tk.Label(root, text="", font=small, bg="#0f172a", fg="#64748b")
ip_lbl.pack(pady=(20, 0))
root.bind("<Escape>", lambda e: root.destroy())   # Esc 退出(无边框窗口)

state = {"emotion": None, "prob": {}, "action": None, "soothe": None,
         "ip_ms": 0, "t": "", "streak": 0}


def apply_state():
    e = state["emotion"]
    if e:
        emo_lbl.config(text=EMO_CN.get(e, e), fg=EMO_COLOR.get(e, "#e2e8f0"))
    p = state["prob"]
    if p:
        for k, (cv, pct) in bars.items():
            v = p.get(k, 0)
            cv.delete("all")
            w = int(cv.winfo_width() * v / 100) if cv.winfo_width() > 1 else 0
            cv.create_rectangle(0, 0, w, 22, fill=EMO_COLOR[k], width=0)
            pct.config(text=f"{v}%")
    a = state["action"]
    if a:
        act_lbl.config(text=f"执行动作: {ACTION_CN.get(a, a)}",
                       fg="#f87171" if a == "出粮" else ("#4ade80" if a == "安抚" else "#38bdf8"))
    if state["soothe"]:
        soothe_lbl.config(text="♪ " + state["soothe"])
    if state["t"]:
        time_lbl.config(text=state["t"])
    if state["streak"]:
        conf_lbl.config(text=f"饥饿确认 {state['streak']}/2 —— 防误触发校验中", fg="#fbbf24")
    else:
        conf_lbl.config(text="")
    if state["ip_ms"]:
        ip_lbl.config(text=f"FPGA 特征提取耗时: {state['ip_ms']} ms")


def parse(line):
    line = line.strip()
    if not line:
        return
    if "[静音]" in line:
        state.update(emotion=None, prob={}, action=None, soothe=None)
        emo_lbl.config(text="(静音段 · 跳过识别)", fg="#475569")
        act_lbl.config(text="执行动作: 保持静默", fg="#38bdf8")
        return
    m = re.match(r"(\d\d:\d\d:\d\d)\s+\S+\s+亲昵(\d+)%\s+饥饿(\d+)%\s+焦虑(\d+)%\s*->\s*(\S+)", line)
    if m:
        state["t"] = m.group(1)
        state["prob"] = {"affection": int(m.group(2)), "food": int(m.group(3)),
                         "isolation": int(m.group(4))}
        state["action"] = m.group(5)
        if state["prob"]["food"] > state["prob"]["affection"] and state["prob"]["food"] > state["prob"]["isolation"]:
            state["emotion"] = "food"
        elif state["prob"]["isolation"] > state["prob"]["affection"]:
            state["emotion"] = "isolation"
        else:
            state["emotion"] = "affection"
        apply_state()
        return
    m = re.search(r"确认 1/2", line)
    if m:
        state["streak"] = 1
        apply_state()
        return
    if "舵机出粮" in line:
        act_lbl.config(text="▶ 执行动作: 投放猫粮(舵机执行)", fg="#f87171")
    elif "安抚语音" in line:
        state["soothe"] = "安抚语音播放中..."
        apply_state()
    m = re.search(r"IP耗时[:=]?\s*(\d+)", line)
    if m:
        state["ip_ms"] = m.group(1)
        apply_state()


def tail_loop():
    if not os.path.exists(LOG):
        ip_lbl.config(text=f"等待日志文件: {LOG}(请先在 PuTTY 里开启 Logging)")
    pos = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    while True:
        try:
            if os.path.exists(LOG):
                sz = os.path.getsize(LOG)
                if sz < pos:  # 日志被重建
                    pos = 0
                if sz > pos:
                    with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(pos)
                        for line in f:
                            parse(line)
                    pos = sz
        except Exception:
            pass
        root.update()
        time.sleep(0.2)


root.after(200, tail_loop)
root.mainloop()
