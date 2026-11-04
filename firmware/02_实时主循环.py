# -*- coding: utf-8 -*-
"""
02_实时主循环.py —— M2/M3: 实时猫叫识别主循环 (PYNQ, 板上运行)
================================================================
数据流: 音频源 -> 1.5s窗口 -> 特征IP -> MLP(集成) -> 触发逻辑 -> 动作

M2阶段: 音频源=SD卡录音文件循环播放模拟实时(先验证整条链路)
M3阶段: 音频源换成I2S麦克风DMA实时采集(改 get_window() 一个函数即可,
        见下方 M3_TODO 注释)

触发规则(与算法侧一致):
  饥饿: 饥饿概率 >= 0.5 且连续2段 -> 出粮
  焦虑: 焦虑概率 >= 0.6 -> 逗猫棒+安抚音频
  静音门限: 窗口RMS < 0.005 -> 不推理
"""
import os
import time
import numpy as np
import wave
from pynq import Overlay, allocate

OVERLAY_BIT = "/home/xilinx/cat_overlay.bit"
WAV_DIR = "/home/xilinx/cat_golden"          # M2模拟用: 测试音频目录
WEIGHT_DIR = "/home/xilinx/cat_weights"      # 模型权重CSV

WIN_LEN = 24000
DIM = 295
DIM_OUT = 1180
THRESH_FOOD = 0.5
THRESH_ISO = 0.6
CONFIRM_FOOD = 2
SILENCE_RMS = 0.005


# ================= MLP集成推理(numpy, 与c_ref/cat_mlp.c同一套权重) =================
def load_mlp(subdir):
    # model1输入295维, model2输入885维: W1文件名不同, 用通配找
    w1_files = [f for f in os.listdir(os.path.join(WEIGHT_DIR, subdir))
                if f.startswith("W1_") and f.endswith(".csv")]
    assert len(w1_files) == 1, f"{subdir}下W1文件异常: {w1_files}"
    W1 = np.loadtxt(os.path.join(WEIGHT_DIR, subdir, w1_files[0]), delimiter=",")
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


MODEL1 = None
MODEL2 = None


def predict(feat295, feat885):
    """集成: 两模型概率平均"""
    global MODEL1, MODEL2
    if MODEL1 is None:
        MODEL1 = load_mlp("model1")
        MODEL2 = load_mlp("model2")
    p1 = mlp_forward(MODEL1, feat295)
    p2 = mlp_forward(MODEL2, feat885)
    return (p1 + p2) / 2


# ================= 特征说明 =================
# IP(v2)直接输出1180维: feat[0:295]=整窗(模型1), feat[295:1180]=3段切片(模型2)
# 板上直接跑完整集成模型(71.4%), 无需PS端再算特征


# ================= 动作执行 =================
def do_feed():
    """M3接GPIO: 触发舵机出粮"""
    print(">>> [动作] 舵机出粮")
    # M3_TODO: gpio.write(1, 1); sleep(0.5); gpio.write(1, 0)


def do_comfort():
    """M3接GPIO: 逗猫棒+播放安抚音频"""
    print(">>> [动作] 安抚: 逗猫棒转动 + 播放音频")
    # M3_TODO: 触发GPIO + I2S播放 /home/xilinx/cat_golden/soothing.wav


# ================= 音频源 =================
def get_window(src):
    """返回一个1.5s窗口的int16数组(24000点)
    M2: 循环播放wav模拟; M3: 换成DMA实时采集"""
    # M3_TODO: 把这里换成 DMA 收满24000点:
    #   dma.recvchannel.transfer(dma_buf)
    #   dma.recvchannel.wait()
    #   return np.array(dma_buf)
    data = src.read_next_window()
    if data is None or len(data) < WIN_LEN:
        return None
    return data[:WIN_LEN]


class WavSource:
    """M2用: 把目录里的wav循环播放(带0.5s静音间隔模拟真实场景)"""
    def __init__(self, wav_dir):
        self.files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
        self.idx = 0
        self.buf = None
        self.pos = 0

    def read_next_window(self):
        while True:
            if self.buf is None or self.pos >= len(self.buf) - WIN_LEN:
                if self.idx >= len(self.files):
                    return None
                with wave.open(os.path.join(WAV_DIR, self.files[self.idx]), "rb") as w:
                    self.buf = np.frombuffer(
                        w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
                self.buf = np.concatenate([np.zeros(8000), self.buf, np.zeros(8000)])
                self.idx += 1
                self.pos = 0
            win = self.buf[self.pos:self.pos + WIN_LEN]
            self.pos += WIN_LEN  # M2用整块步进; 真实流式用滑窗(见M3_TODO)
            return win.astype(np.int16)


# ================= 主循环 =================
def main():
    global MODEL1, MODEL2
    ol = Overlay(OVERLAY_BIT)
    # 找到特征IP
    ip = None
    for name in ol.ip_dict:
        if name.startswith("cat_feat_ip"):
            ip = getattr(ol, name)
            break
    if ip is None:
        raise RuntimeError("overlay里没找到特征IP")

    pcm_buf = allocate(shape=(WIN_LEN,), dtype=np.int32)
    feat_buf = allocate(shape=(DIM_OUT,), dtype=np.float32)  # IP输出1180维
    MODEL1 = load_mlp("model1")
    MODEL2 = load_mlp("model2")
    print("模型已加载(集成), 音频源: %s" % WAV_DIR)

    src = WavSource(WAV_DIR)
    food_streak = 0
    print("实时识别启动 (M2模拟模式)...")
    while True:
        pcm = get_window(src)
        if pcm is None:
            print("音频放完, 循环重新开始")
            src = WavSource(WAV_DIR)
            continue

        # 静音门限
        rms = np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2))
        if rms < SILENCE_RMS:
            print(f"[静音] rms={rms:.4f} 跳过")
            continue

        # 驱动特征IP (输出: 295整窗 + 885分段)
        np.copyto(pcm_buf, pcm.astype(np.int32))
        pa = int(pcm_buf.physical_address)
        ip.write(0x10, pa & 0xFFFFFFFF)
        ip.write(0x14, pa >> 32)
        fa = int(feat_buf.physical_address)
        ip.write(0x1c, fa & 0xFFFFFFFF)
        ip.write(0x20, fa >> 32)
        ip.write(0x00, 0x1)
        _t0 = time.perf_counter()
        while not (ip.read(0x00) & 0x2):
            if time.perf_counter() - _t0 > 60:
                raise RuntimeError("IP 60秒未完成: 检查地址映射/时钟")
        feat = np.array(feat_buf, dtype=np.float64)
        feat295 = feat[:DIM]
        feat885 = feat[DIM:DIM_OUT]

        # 集成推理: 两模型概率平均 (CV 71.4%)
        p = predict(feat295, feat885)
        name = ["亲昵", "饥饿", "焦虑"][int(p.argmax())]
        print(f"[{time.strftime('%H:%M:%S')}] {name} 亲昵{p[0]:.2f}/饥饿{p[1]:.2f}/焦虑{p[2]:.2f}")

        # 触发逻辑
        if p[1] >= THRESH_FOOD:
            food_streak += 1
            if food_streak >= CONFIRM_FOOD:
                do_feed()
                food_streak = 0
        else:
            food_streak = 0
        if p[2] >= THRESH_ISO:
            do_comfort()


if __name__ == "__main__":
    main()
