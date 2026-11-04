# -*- coding: utf-8 -*-
"""
00_生成矩阵.py —— 特征规格固定矩阵生成器(纯numpy部署与HLS共用的数据文件)
================================================================
本脚本用 librosa 1.0.0 一次性生成特征提取所需的全部固定矩阵/系数,
导出为 CSV, 供:
  1. features_numpy.py 纯numpy特征提取(板端PS)直接加载
  2. HLS 队友直接 #include 或初始化表(无需在FPGA上重算mel/DCT公式)

生成物:
  - mel_128x1025.csv   : Mel滤波器组矩阵(Slaney norm), 128行 x 1025列
  - dct_40x128.csv     : DCT-II(ortho)矩阵, 40行 x 128列
  - hann_400.csv       : 周期性Hann窗(400点, fftbins=True)
  - hann_2048.csv      : 周期性Hann窗(2048点)
  - savgol_d1.csv      : delta一阶滤波: 行=每帧位置, 列=卷积系数(151帧 x 151稀疏→存9列+偏移)
  - savgol_d2.csv      : delta二阶同上
运行: py -3.12 00_生成矩阵.py
"""
import os
import numpy as np
import scipy.signal
import scipy.fft
import librosa

OUT = os.path.dirname(os.path.abspath(__file__))

SR = 16000
N_FFT = 2048
WIN = 400        # 25ms MFCC窗
HOP = 160        # 10ms MFCC帧移
N_MELS = 128
N_MFCC = 40

# 固定1.5s窗口 => MFCC帧数恒定 = 1 + 24000/160 = 151
NFRAMES_MFCC = 151

# ---------- 1. Mel滤波器组 (Slaney, 与librosa.filters.mel一致) ----------
mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                fmin=0.0, fmax=SR / 2, htk=False, norm="slaney")
np.savetxt(os.path.join(OUT, "mel_128x1025.csv"), mel_basis, delimiter=",",
           fmt="%.10e", header="mel filterbank 128x1025, slaney norm, sr=16000")
print(f"mel_128x1025.csv  {mel_basis.shape}  每行非零数: {(mel_basis>0).sum(axis=1).min()}~{(mel_basis>0).sum(axis=1).max()}")

# ---------- 2. DCT-II(ortho)矩阵 40x128 ----------
N_MEL_BINS = N_MELS
dct = np.zeros((N_MFCC, N_MEL_BINS))
for k in range(N_MFCC):
    dct[k] = np.cos(np.pi * k * (2 * np.arange(N_MEL_BINS) + 1) / (2 * N_MEL_BINS))
dct *= np.sqrt(2.0 / N_MEL_BINS)
dct[0] /= np.sqrt(2.0)
# 与 scipy.fft.dct(type=2, norm='ortho') 对拍
x = np.random.randn(N_MEL_BINS)
err = np.abs(scipy.fft.dct(x, type=2, norm="ortho")[:N_MFCC] - dct @ x).max()
print(f"dct_40x128.csv    {dct.shape}  与scipy对拍最大误差: {err:.2e}")
np.savetxt(os.path.join(OUT, "dct_40x128.csv"), dct, delimiter=",",
           fmt="%.10e", header="DCT-II ortho 40x128")

# ---------- 3. 周期性Hann窗 (scipy get_window fftbins=True) ----------
hann400 = scipy.signal.get_window("hann", WIN, fftbins=True)
hann2048 = scipy.signal.get_window("hann", N_FFT, fftbins=True)
np.savetxt(os.path.join(OUT, "hann_400.csv"), hann400, delimiter=",", fmt="%.10e")
np.savetxt(os.path.join(OUT, "hann_2048.csv"), hann2048, delimiter=",", fmt="%.10e")
print(f"hann_400/2048.csv  周期窗系数  (HLS注意: MFCC窗=400点居中嵌入2048点帧)")

# ---------- 4. Savitzky-Golay delta 系数 (librosa 1.0 的 delta 实现) ----------
# 用单位冲激探测法得到 savgol_filter(width=9, polyorder=order, deriv=order, mode='interp')
# 的完整线性算子, 提取"每帧位置对应的卷积系数", 存成 HLS 可直接用的表
def probe_savgol(n_frames, order):
    W, P = 9, order
    T = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        imp = np.zeros(n_frames)
        imp[i] = 1.0
        T[:, i] = scipy.signal.savgol_filter(imp, window_length=W, polyorder=P,
                                             deriv=order, mode="interp")
    return T

for order, name in [(1, "savgol_d1"), (2, "savgol_d2")]:
    T = probe_savgol(NFRAMES_MFCC, order)
    # 存稠密矩阵(151x151), HLS侧可离线转成稀疏/边界表
    np.savetxt(os.path.join(OUT, f"{name}.csv"), T, delimiter=",", fmt="%.10e",
               header=f"savgol delta order={order}, rows=输出帧, cols=输入帧")
    # 检查内部是否平移不变(应为9抽头FIR)
    interior = T[10, 6:15]  # 第10行, 邻域4
    print(f"{name}.csv         151x151  内部9抽头系数: {np.round(interior, 4)}  对称: {np.allclose(interior, interior[::-1])}")

# 分段特征(0.5s切片=8000样本=51帧)用的savgol矩阵
NFRAMES_SEG = 1 + 8000 // HOP   # 51
for order, name in [(1, "savgol_d1_51"), (2, "savgol_d2_51")]:
    T = probe_savgol(NFRAMES_SEG, order)
    np.savetxt(os.path.join(OUT, f"{name}.csv"), T, delimiter=",", fmt="%.10e",
               header=f"savgol delta order={order}, 51帧(0.5s切片)")
    print(f"{name}.csv   51x51  用于分段特征")

print("\n全部生成完毕 ->", OUT)
