#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_m2_full.py - LED演示: 实时识别 + LED动作指示 (v5广播overlay)
====================================================================
在02_m2_servo.py基础上增加喇叭安抚播放:
  特征IP寄存器:  0x00 ap_ctrl, 0x2000 特征(1180 float), 0x10000 PCM输入
  PWM IP寄存器:  0x00 ap_ctrl, 0x10 脉宽(0.1us), 0x18 周期(0.1us)
  I2S IP寄存器:  0x00 ap_ctrl, 0x10起12000个寄存器(每寄存器2个int16样本)
  安抚动作: soothing.wav平铺到24000点 -> 写I2S IP -> 播放2遍(共3秒)
接线: 舵机信号->JA1; 喇叭BCLK->JA2 WS->JA3 DIN->JA4 (VIN接5V, 共地)
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
PWM_BASE = 0x40030000        # v5i构建(特征IP 128K段)后的映射, 以hwh核对
I2S_BASE = 0x40020000        # 同上
IP_SIZE = 0x40000
OFF_AP = 0x00000
OFF_FEAT = 0x02000
OFF_PCM = 0x10000
WIN_LEN = 24000
DIM = 295
DIM_OUT = 1180
TIMEOUT = 60

PWM_AP = 0x00
PWM_WIDTH = 0x10
PWM_PERIOD = 0x18
PWM_POS_IDLE = 0         # LED模式: 静止=LED0灭
PWM_POS_DROP = 20000     # LED模式: 出粮=LED0全亮(100%占空)

I2S_AP = 0x00
I2S_DATA = 0x10          # 12000个int32寄存器 = 24000样本
I2S_NREG = 12000
SOOTHE_ROUNDS = 2        # 安抚播2遍, 每遍1.5秒

GOLDEN_DIR = "/home/xilinx/cat_golden"
BIT_PATH = "/home/xilinx/cat_overlay.bit"
WEIGHT_DIR = "/home/xilinx/cat_weights"
STATUS_PATH = "/opt/m1/status.json"

THRESH_FOOD = 0.5
THRESH_ISO = 0.6
CONFIRM_FOOD = 2
SILENCE_RMS = 0.005

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
        print("[1/6] 比特流已加载 (/dev/fpga0)")
        return
    except Exception:
        pass
    os.system("mkdir -p /lib/firmware")
    open("/lib/firmware/cat_overlay.bin", "wb").write(bin_data)
    with open("/sys/class/fpga_manager/fpga0/firmware", "w") as f:
        f.write("cat_overlay.bin")
    print("[1/6] 比特流已加载 (sysfs)")


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
    print("LED演示: 实时识别 + LED动作指示 (v5广播overlay)")
    print("=" * 56)

    print("[1/6] 加载比特流...")
    load_bitstream(bit_to_bin(BIT_PATH))
    time.sleep(2)

    print("[2/6] 映射寄存器 + 加载模型...")
    reg = map_mem(IP_BASE, IP_SIZE)
    pwm = map_mem(PWM_BASE, 0x10000)
    i2s = map_mem(I2S_BASE, 0x10000)
    pcm_regs = np.frombuffer(reg, dtype=np.int32, count=12000, offset=OFF_PCM)
    feat_regs = np.frombuffer(reg, dtype=np.float32, count=DIM_OUT, offset=OFF_FEAT)
    m1 = load_mlp("model1")
    m2 = load_mlp("model2")

    def w(off, val):
        struct.pack_into("<I", reg, off, val & 0xFFFFFFFF)

    def r(off):
        return struct.unpack_from("<I", reg, off)[0]

    def pwm_w(off, val):
        struct.pack_into("<I", pwm, off, val & 0xFFFFFFFF)

    def i2s_w(off, val):
        struct.pack_into("<I", i2s, off, val & 0xFFFFFFFF)

    def i2s_r(off):
        return struct.unpack_from("<I", i2s, off)[0]

    # ---- 初始化舵机 ----
    print("[3/6] 初始化舵机 (周期20ms, 初始堵粮位)...")
    pwm_w(PWM_PERIOD, 200000)
    pwm_w(PWM_WIDTH, PWM_POS_IDLE)
    pwm_w(PWM_AP, 0x1)          # 关键: PWM模块是while(1)死循环, 必须ap_start才运行
    time.sleep(1)
    print("      舵机就绪")

    # ---- 准备安抚音频 ----
    print("[4/6] 加载安抚音频...")
    soothe_path = os.path.join(GOLDEN_DIR, "soothing.wav")
    if os.path.exists(soothe_path):
        with wave.open(soothe_path, "rb") as wv:
            pcm_s = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16)
        if wv.getframerate() != 16000:
            print("      警告: soothing.wav采样率非16k, 按16k重采样")
            idx = np.linspace(0, len(pcm_s) - 1, int(len(pcm_s) * 16000 / wv.getframerate()))
            pcm_s = np.interp(idx, np.arange(len(pcm_s)), pcm_s).astype(np.int16)
        pcm_s = np.resize(pcm_s, WIN_LEN)   # 平铺/裁剪到24000点
    else:
        print("      未找到soothing.wav, 生成柔和提示音代替")
        t = np.arange(WIN_LEN) / 16000.0
        pcm_s = (6000 * np.sin(2 * np.pi * 523.25 * t) +
                 4000 * np.sin(2 * np.pi * 659.25 * t)) * (
                     np.minimum(1.0, t * 4) * np.minimum(1.0, (WIN_LEN / 16000 - t) * 4))
        pcm_s = pcm_s.astype(np.int16)
    p32 = pcm_s.astype(np.int32)
    packed_soothe = (p32[0::2] & 0xFFFF) | ((p32[1::2] & 0xFFFF) << 16)
    for k in range(I2S_NREG):
        i2s_w(I2S_DATA + 4 * k, int(packed_soothe[k]))
    print("      安抚音频就绪 (24000点/1.5秒)")

    def play_soothe():
        for k in range(SOOTHE_ROUNDS):
            i2s_w(I2S_AP, 0x0)
            i2s_w(I2S_AP, 0x1)          # ap_start
            t0 = time.time()
            while not (i2s_r(I2S_AP) & 0x2):
                if time.time() - t0 > 8:
                    print("      ! 喇叭IP超时")
                    return
            i2s_w(I2S_AP, 0x0)
        print("      (安抚语音播放完毕)")

    def feed_action():
        print(">>> [动作] 舵机出粮!")
        pwm_w(PWM_WIDTH, PWM_POS_DROP)     # 转开缺口
        time.sleep(0.6)
        pwm_w(PWM_WIDTH, PWM_POS_IDLE)     # 转回堵住
        time.sleep(0.3)

    files = sorted(f for f in os.listdir(GOLDEN_DIR) if f.endswith("_wav.wav"))
    print("[5/6] 开始实时识别...")
    print()

    food_streak = 0
    cycle = 0
    while True:
        cycle += 1
        print("-" * 56)
        print("第 %d 轮演示开始" % cycle)
        for f in files:
            base = f.replace("_wav.wav", "")
            with wave.open(os.path.join(GOLDEN_DIR, f), "rb") as wv:
                pcm = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16)
            pcm = pcm[:WIN_LEN]

            rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2)))
            if rms < SILENCE_RMS:
                st = {"time": time.strftime("%H:%M:%S"), "status": "silence",
                      "emotion": "unknown", "emotion_cn": "未知",
                      "prob": {"affection": 0, "food": 0, "isolation": 0},
                      "action": "none", "action_cn": "不动作",
                      "soothing": None, "ip_time_ms": 0, "confirm_streak": food_streak}
                write_status(st)
                print("[静音] 跳过")
                time.sleep(0.3)
                continue

            p32 = pcm.astype(np.int32)
            packed = (p32[0::2] & 0xFFFF) | ((p32[1::2] & 0xFFFF) << 16)
            np.copyto(pcm_regs, packed)
            w(OFF_AP, 0x1)
            t0 = time.time()
            while not (r(OFF_AP) & 0x2):
                if time.time() - t0 > TIMEOUT:
                    raise RuntimeError("IP超时")
            dt_ms = int((time.time() - t0) * 1000)

            feat = np.array(feat_regs, dtype=np.float64)
            p = (mlp_forward(m1, feat[:DIM]) + mlp_forward(m2, feat[DIM:DIM_OUT])) / 2
            cls = ["affection", "food", "isolation"][int(p.argmax())]

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
                soothing = "正在播放安抚语音 soothing.wav (喇叭)"

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
                feed_action()
            elif action == "comfort":
                print(">>> [动作] 播放安抚语音 (MAX98357A喇叭)")
                play_soothe()
            if food_streak == 1:
                print("    (饥饿确认 1/2, 等待下一段...)")

            if True:
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
