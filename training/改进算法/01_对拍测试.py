# -*- coding: utf-8 -*-
"""
01_对拍测试.py —— 纯numpy特征 vs librosa特征 逐位对拍
====================================================
验证 features_numpy.py 与 librosa 1.0.0 在固定1.5s窗口输入下完全一致。
通过标准: 每个295维特征分量的最大绝对误差 < 1e-3。

测试集: 随机噪声(多种长度) + 全量440段真实猫叫
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_numpy as fn
import librosa

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cat_dataset")
CLASSES = ["affection", "food", "isolation"]


def extract_librosa(y):
    """librosa 1.0.0 参考实现: 与训练端一致 + 固定窗口 + 全零帧质心=0"""
    y = fn.fixed_window(y).astype(np.float64)
    mfcc = librosa.feature.mfcc(y=y, sr=fn.SR, n_mfcc=fn.N_MFCC,
                                win_length=fn.WIN, hop_length=fn.HOP)
    d1 = librosa.feature.delta(mfcc)                  # width=9 order=1 mode='interp'
    d2 = librosa.feature.delta(mfcc, order=2)
    cent = librosa.feature.spectral_centroid(y=y, sr=fn.SR)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cent = np.nan_to_num(cent, nan=0.0)               # 全零帧定义0 (与numpy版一致)
    per_mfcc = np.concatenate([fn._stats(mfcc[i]) for i in range(fn.N_MFCC)])
    return np.concatenate([per_mfcc, d1.mean(axis=1), d2.mean(axis=1),
                           fn._stats(cent), fn._stats(rms), fn._stats(zcr)])


def compare(name, y):
    a = fn.extract_features_from_wave(y)
    b = extract_librosa(y)
    err = np.abs(a - b)
    return err.max()


def main():
    np.random.seed(0)
    results = []

    # --- 随机噪声, 各种长度(重点测窗口选择/补零/帧数边界) ---
    for i, n in enumerate([100, 1600, 10000, 15999, 16000, 23999, 24000, 24001,
                           30000, 48000, 64000]):
        y = np.random.randn(n) * 0.01
        results.append((f"随机噪声 len={n}", compare("noise", y)))

    # --- 全量真实音频 ---
    files = []
    for c in CLASSES:
        for f in os.listdir(os.path.join(DATASET, c)):
            files.append(os.path.join(DATASET, c, f))
    worst = 0.0
    worst_name = ""
    for f in files:
        y, _ = librosa.load(f, sr=fn.SR)
        e = compare(os.path.basename(f), y)
        if e > worst:
            worst, worst_name = e, os.path.basename(f)
        results.append((f, e))

    print(f"共 {len(results)} 组对拍")
    print(f"最差误差: {worst:.3e}  ({worst_name})")
    print(f"误差>1e-3 的组数: {sum(1 for _, e in results if e > 1e-3)}")
    for name, e in results[:11]:
        print(f"  {name}: {e:.3e}")
    ok = all(e < 1e-3 for _, e in results)
    print("\n对拍结论:", "PASS ✓ 纯numpy版与librosa版一致" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
