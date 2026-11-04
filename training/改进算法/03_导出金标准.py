# -*- coding: utf-8 -*-
"""
03_导出金标准.py —— HLS对拍用金标准数据
========================================
从440段中按类别比例抽40段, 每段导出:
  pcm_16k_int16.csv   固定1.5s窗口后的24000个16bit样本(特征提取的精确输入)
  features_295.csv    纯numpy特征(与librosa已对拍一致, HLS实现的参考值)
  probs_3.csv         最终float模型输出的3类概率
  manifest.csv        清单(文件名/类别/概率)
HLS验收标准: 特征分量最大误差<1e-3, 概率差<0.01
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_ref as fr
import features_numpy as fn
from mlp_numpy import load_ensemble, predict_ensemble

OUT = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(OUT, "golden")
os.makedirs(GOLDEN, exist_ok=True)
DATASET = os.path.join(os.path.dirname(OUT), "cat_dataset")
CLASSES = ["affection", "food", "isolation"]
# 按类别比例抽40段: 12/8/20
N_PER = {"affection": 12, "food": 8, "isolation": 20}
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    m1, m2 = load_ensemble()
    manifest = []
    for cls in CLASSES:
        files = sorted(os.listdir(os.path.join(DATASET, cls)))
        picks = rng.choice(len(files), N_PER[cls], replace=False)
        for i in picks:
            f = files[i]
            y = fr.load_wav(os.path.join(DATASET, cls, f))
            yw = fr.fixed_window(y)
            pcm = np.clip(np.round(yw * 32768), -32768, 32767).astype(np.int16)
            # 特征从PCM回读(与板上输入完全一致), 保证 golden 链路自洽可复现
            yq = pcm.astype(np.float64) / 32768.0
            feat295 = fn.extract_features_from_wave(yq)
            feat885 = fn.extract_features_segmental(yq)
            _, prob = predict_ensemble(feat295, feat885, m1, m2)
            base = os.path.splitext(f)[0]
            np.savetxt(os.path.join(GOLDEN, base + "_pcm.csv"), pcm, fmt="%d")
            np.savetxt(os.path.join(GOLDEN, base + "_features295.csv"), feat295, fmt="%.9e")
            np.savetxt(os.path.join(GOLDEN, base + "_features885.csv"), feat885, fmt="%.9e")
            np.savetxt(os.path.join(GOLDEN, base + "_probs.csv"), prob, fmt="%.9e")
            manifest.append([f, cls, int(pcm.shape[0]), *[round(float(p), 6) for p in prob]])
            print(f"  {cls}/{f}  prob={np.round(prob,3)}")
    hdr = "file,class,n_samples,p_affection,p_food,p_isolation"
    with open(os.path.join(GOLDEN, "manifest.csv"), "w", encoding="utf-8") as fh:
        fh.write(hdr + "\n")
        for row in manifest:
            fh.write(",".join(map(str, row)) + "\n")
    print(f"\n金标准导出完成: {len(manifest)}段 -> {GOLDEN} (含pcm/features295/features885/probs)")


if __name__ == "__main__":
    main()
