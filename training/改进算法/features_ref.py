# -*- coding: utf-8 -*-
"""
features_ref.py —— librosa 1.0.0 参考特征实现(训练端/对拍端共用)
================================================================
两条管线:
  A. 固定窗口版(部署规格, 板上/HLS按此实现):
       fixed_window 1.5s -> 295维特征, 无静音裁剪
  B. 原版(legacy, 与01-训练代码/训练脚本.py一致):
       trim(top_db=20) -> 补零到0.1s -> 变长特征

与纯numpy版(features_numpy.py)的唯一有意差异(已在对拍中验证一致):
  全零帧的谱质心定义为 0 (librosa原生会算出nan)
"""
import numpy as np
import librosa

SR = 16000
N_MFCC = 40
WIN = 400
HOP = 160

FIXED_LEN = 24000     # 1.5s
BLOCK = 1600          # 能量窗选择步进 0.1s
NWIN_BLOCKS = 15


def fixed_window(y, win_len=FIXED_LEN):
    """固定1.5s窗口: 不足补零(尾部); 超长取0.1s块能量和最大的连续1.5s"""
    n = len(y)
    if n < win_len:
        return np.pad(y, (0, win_len - n))
    n_blocks = n // BLOCK
    if n_blocks < NWIN_BLOCKS:
        return np.pad(y[:NWIN_BLOCKS * BLOCK], (0, max(0, win_len - n)))
    e = np.array([np.sum(y[i * BLOCK:(i + 1) * BLOCK] ** 2) for i in range(n_blocks)])
    csum = np.convolve(e, np.ones(NWIN_BLOCKS), mode="valid")
    start = int(np.argmax(csum)) * BLOCK
    return y[start:start + win_len]


def _stats(x):
    p = np.percentile(x, [10, 50, 90])
    return np.concatenate([p, [x.mean(), x.std()]])


def _extract_from_wave(y):
    """295维特征(与训练脚本.py相同的组装顺序)"""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                win_length=WIN, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    cent = np.nan_to_num(librosa.feature.spectral_centroid(y=y, sr=SR)[0], nan=0.0)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    per_mfcc = np.concatenate([_stats(mfcc[i]) for i in range(mfcc.shape[0])])
    return np.concatenate([per_mfcc, d1.mean(axis=1), d2.mean(axis=1),
                           _stats(cent), _stats(rms), _stats(zcr)])


def extract_fixed(y):
    """管线A: 固定窗口版(部署规格)"""
    return _extract_from_wave(fixed_window(y).astype(np.float64))


def extract_legacy(y):
    """管线B: 原版(trim + 变长)"""
    y = librosa.effects.trim(y, top_db=20)[0]
    if len(y) < 1600:
        y = np.pad(y, (0, 1600 - len(y)))
    return _extract_from_wave(y.astype(np.float64))


def load_wav(path):
    return librosa.load(path, sr=SR)[0]


def augment_food(y):
    """饥饿类增强(与原版一致): 原样+变速±1.5半音+拉伸0.9/1.1+前4版加噪"""
    vers = [y]
    vers.append(librosa.effects.pitch_shift(y, sr=SR, n_steps=1.5))
    vers.append(librosa.effects.pitch_shift(y, sr=SR, n_steps=-1.5))
    vers.append(librosa.effects.time_stretch(y, rate=0.9))
    vers.append(librosa.effects.time_stretch(y, rate=1.1))
    for v in vers[:4]:
        vers.append(v + 0.005 * np.random.randn(len(v)))
    return vers
