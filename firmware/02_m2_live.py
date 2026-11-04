#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_m2_live.py - M2实时演示主循环(裸机版, 无pynq)
================================================
流程: 循环播放40段测试音频 -> 每段=1.5s窗口
      静音门限(RMS<0.005) -> 特征IP(v3寄存器) -> MLP集成 -> 触发逻辑
同步输出:
  1. 终端实时打印(录视频用)
  2. status.json(显示屏UI读取, 字段与《给显示屏同学-提示词.txt》约定一致)
运行: sudo python3 /opt/m1/02_m2_live.py
"""
import os
import sys
import time
import json
import wave
import struct
import mmap
import numpy as np

IP_BASE = 0x40000000
IP_SIZE = 0x40000
OFF_AP = 0x00000
OFF_FEAT = 0x02000
OFF_PCM = 0x10000
WIN_LEN = 24000
DIM = 295
DIM_OUT = 1180
TIMEOUT = 60

GOLDEN_DIR = "/home/xilinx/cat_golden"
BIT_PATH = "/home/xilinx/cat_overlay.bit"
WEIGHT_DIR = "/home/xilinx/cat_weights"
STATUS_PATH = "/opt/m1/status.json"

THRESH_FOOD = 0.5
THRESH_ISO = 0.6
CONFIRM_FOOD = 2
SILENCE_RMS = 0.005
GAP_WINDOWS = 1          # 每段之间插入的静音窗口数(模拟真实间隔)

EMOJI = {"affection": "😊", "food": "🍽️", "isolation": "😿", "unknown": "❓"}
ACTION_CN = {"none": "不动作", "feed": "出粮", "comfort": "安抚"}


def map_mem(phys, size):
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, size, mmap.MAP_SHARED,
                  mmap.PROT_READ | mmap.PROT_WRITE, offset=phys)
    os.close(fd)
    return m


def bit_to_bin(bit_path):
    data = open(bit_path, "rb").read()
    for i in range(len(data) - 3):
        if data[i] == 0xAA and data[i + 1] == 0x99 and \
           data[i + 2] == 0x55 and data[i + 3] == 0x66:
            d = bytearray(data[i:])
            d[0::4], d[3::4] = d[3::4], d[0::4]
            d[1::4], d[2::4] = d[2::4], d[1::4]
            return bytes(d)
    raise RuntimeError("sync not found")


def load_bitstream(bin_data):
    try:
        fd = os.open("/dev/fpga0", os.O_WRONLY)
        os.write(fd, bin_data)
        os.close(fd)
        print("[1/4] 比特流已加载 (/dev/fpga0)")
        return
    except Exception:
        pass
    os.system("mkdir -p /lib/firmware")
    open("/lib/firmware/cat_overlay.bin", "wb").write(bin_data)
    with open("/sys/class/fpga_manager/fpga0/firmware", "w") as f:
        f.write("cat_overlay.bin")
    print("[1/4] 比特流已加载 (sysfs)")


def load_mlp(subdir):
    w1f = [f for f in os.listdir(os.path.join(WEIGHT_DIR, subdir))
           if f.startswith("W1_") and f.endswith(".csv")][0]
    W1 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, w1f), delimiter=",")
    b1 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "b1_512.csv"), delimiter=",")
    W2 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "W2_256x512.csv"), delimiter=",")
    b2 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "b2_256.csv"), delimiter=",")
    W3 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "W3_3x256.csv"), delimiter=",")
    b3 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "b3_3.csv"), delimiter=",")
    mean = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "scaler_mean.csv"), delimiter=",")
    scale = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, "scaler_scale.csv"), delimiter=",")
    return (W1, b1, W2, b2, W3, b3, mean, scale)


def mlp_forward(m, x):
    W1, b1, W2, b2, W3, b3, mean, scale = m
    x = (x - mean) / scale
    h = np.maximum(x @ W1.T + b1, 0)
    h = np.maximum(h @ W2.T + b2, 0)
    logits = h @ W3.T + b3
    e = np.exp(logits - logits.max())
    return e / e.sum()


def write_status(st):
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False)
    os.rename(tmp, STATUS_PATH)


def main():
    print("=" * 56)
    print("M2 实时演示 (裸机版): 猫叫情绪识别")
    print("=" * 56)

    print("[1/4] 加载比特流...")
    load_bitstream(bit_to_bin(BIT_PATH))
    time.sleep(2)

    print("[2/4] 映射寄存器 + 加载模型...")
    reg = map_mem(IP_BASE, IP_SIZE)
    pcm_regs = np.frombuffer(reg, dtype=np.int32, count=12000, offset=OFF_PCM)
    feat_regs = np.frombuffer(reg, dtype=np.float32, count=DIM_OUT, offset=OFF_FEAT)
    m1 = load_mlp("model1")
    m2 = load_mlp("model2")

    def ip_w(off, val):
        struct.pack_into("<I", reg, off, val & 0xFFFFFFFF)

    def ip_r(off):
        return struct.unpack_from("<I", reg, off)[0]

    files = sorted(f for f in os.listdir(GOLDEN_DIR) if f.endswith("_wav.wav"))
    print("[3/4] 开始实时识别 (循环播放 %d 段音频, Ctrl+C 停止)..." % len(files))
    print()

    food_streak = 0
    cycle = 0
    while True:
        cycle += 1
        print("-" * 56)
        print("第 %d 轮演示开始" % cycle)
        for f in files:
            base = f.replace("_wav.wav", "")
            with wave.open(os.path.join(GOLDEN_DIR, f), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pcm = pcm[:WIN_LEN]

            # ---- 静音门限 ----
            rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2)))
            if rms < SILENCE_RMS:
                st = {"time": time.strftime("%H:%M:%S"), "status": "silence",
                      "emotion": "unknown", "emotion_cn": "未知",
                      "prob": {"affection": 0, "food": 0, "isolation": 0},
                      "action": "none", "action_cn": "不动作",
                      "soothing": None, "ip_time_ms": 0, "confirm_streak": 0}
                write_status(st)
                print("[静音] 跳过")
                time.sleep(0.3)
                continue

            # ---- 特征IP (v3寄存器通道) ----
            p32 = pcm.astype(np.int32)
            packed = (p32[0::2] & 0xFFFF) | ((p32[1::2] & 0xFFFF) << 16)
            np.copyto(pcm_regs, packed)
            ip_w(OFF_AP, 0x1)
            t0 = time.time()
            while not (ip_r(OFF_AP) & 0x2):
                if time.time() - t0 > TIMEOUT:
                    raise RuntimeError("IP超时")
            dt_ms = int((time.time() - t0) * 1000)

            feat = np.array(feat_regs, dtype=np.float64)
            p = (mlp_forward(m1, feat[:DIM]) + mlp_forward(m2, feat[DIM:DIM_OUT])) / 2
            cls = ["affection", "food", "isolation"][int(p.argmax())]

            # ---- 触发逻辑 ----
            soothing = None
            action = "none"
            if p[1] >= THRESH_FOOD:
                food_streak += 1
                if food_streak >= CONFIRM_FOOD:
                    action = "feed"
                    food_streak = 0
            else:
                food_streak = 0
            if p[2] >= THRESH_ISO:
                action = "comfort"
                soothing = "正在播放安抚语音 01.mp3"

            # ---- 状态输出 ----
            st = {"time": time.strftime("%H:%M:%S"), "status": "listening",
                  "emotion": cls, "emotion_cn": {"affection": "亲昵", "food": "饥饿",
                                                 "isolation": "焦虑"}[cls],
                  "prob": {"affection": round(float(p[0]), 4),
                           "food": round(float(p[1]), 4),
                           "isolation": round(float(p[2]), 4)},
                  "action": action, "action_cn": ACTION_CN[action],
                  "soothing": soothing, "ip_time_ms": dt_ms,
                  "confirm_streak": food_streak}
            write_status(st)

            line = "%s  %s%s  亲昵%.0f%% 饥饿%.0f%% 焦虑%.0f%%  ->  %s" % (
                st["time"], st["emotion_cn"], EMOJI[cls],
                p[0] * 100, p[1] * 100, p[2] * 100, st["action_cn"])
            print(line)
            if action == "feed":
                print(">>> [动作] 舵机出粮! (演示: 实际GPIO在M3接入)")
            elif action == "comfort":
                print(">>> [动作] 播放安抚语音 (喇叭M3接入, 当前屏幕展示)")
            if food_streak == 1:
                print("    (饥饿确认 1/2, 等待下一段...)")

            # 间隔静音(模拟真实场景)
            if GAP_WINDOWS:
                st = {"time": time.strftime("%H:%M:%S"), "status": "silence",
                      "emotion": "unknown", "emotion_cn": "未知",
                      "prob": {"affection": 0, "food": 0, "isolation": 0},
                      "action": "none", "action_cn": "不动作",
                      "soothing": None, "ip_time_ms": 0, "confirm_streak": food_streak}
                write_status(st)
                time.sleep(0.5)

        print("第 %d 轮结束, 重新开始..." % cycle)


if __name__ == "__main__":
    main()
