# -*- coding: utf-8 -*-
"""
09_最终模型.py —— 集成模型(2xMLP概率平均)全量训练 + float/int8导出 + 最终阈值
==============================================================================
模型1: 295维特征, 平衡权重, 全类增强训练集
模型2: 885维分段特征, 平衡权重, food增强训练集
推理: prob = (prob1 + prob2) / 2
产物:
  weights_float_final/model1|model2/   float权重+scaler
  weights_int8_final/model1|model2/    int8权重+layer_scales.json
  final_meta.json                      最终配置/阈值/CV指标
"""
import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "_cache")
CLASSES = ["affection", "food", "isolation"]
SEED = 42


def fit_balanced(X, y):
    model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                          random_state=SEED, early_stopping=True,
                          validation_fraction=0.15, learning_rate_init=0.0005)
    w = {gi: 1.0 / (y == gi).sum() for gi in np.unique(y)}
    sw = np.array([w[v] for v in y])
    model.fit(X, y, sample_weight=sw)
    return model


def export_weights(model, scaler, outdir, quantize=False, nbits=127, Xs=None):
    os.makedirs(outdir, exist_ok=True)
    scales_json = {}
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        if quantize:
            s = np.abs(W).max() / nbits
            Wq = np.clip(np.round(W / s), -nbits, nbits).astype(np.int16)
            np.savetxt(os.path.join(outdir, f"W{i+1}_{Wq.shape[1]}x{Wq.shape[0]}.csv"),
                       Wq.T, delimiter=",", fmt="%d")
            scales_json[f"w{i+1}_scale"] = float(s)
        else:
            np.savetxt(os.path.join(outdir, f"W{i+1}_{W.shape[1]}x{W.shape[0]}.csv"),
                       W.T, delimiter=",", fmt="%.9e")
        np.savetxt(os.path.join(outdir, f"b{i+1}_{b.shape[0]}.csv"), b,
                   delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(outdir, "scaler_mean.csv"), scaler.mean_, delimiter=",",
               fmt="%.9e")
    np.savetxt(os.path.join(outdir, "scaler_scale.csv"), scaler.scale_, delimiter=",",
               fmt="%.9e")
    if quantize and Xs is not None:
        h1 = np.maximum(Xs @ model.coefs_[0] + model.intercepts_[0], 0)
        h2 = np.maximum(h1 @ model.coefs_[1] + model.intercepts_[1], 0)
        scales_json["x_scale"] = float(np.abs(Xs).max() / nbits)
        scales_json["h1_scale"] = float(np.abs(h1).max() / nbits)
        scales_json["h2_scale"] = float(np.abs(h2).max() / nbits)
        with open(os.path.join(outdir, "layer_scales.json"), "w") as f:
            json.dump(scales_json, f, indent=2)


def main():
    # ---- 模型1: 295维 + 全类增强 ----
    X1 = np.load(os.path.join(CACHE, "feat_allaug_X.npy"))
    y1 = np.load(os.path.join(CACHE, "feat_allaug_y.npy"))
    sc1 = StandardScaler().fit(X1)
    m1 = fit_balanced(sc1.transform(X1), y1)

    # ---- 模型2: 885维分段 + food增强 ----
    X2 = np.load(os.path.join(CACHE, "feat_seg_X.npy"))
    y2 = np.load(os.path.join(CACHE, "feat_seg_y.npy"))
    sc2 = StandardScaler().fit(X2)
    m2 = fit_balanced(sc2.transform(X2), y2)

    # ---- 导出 float ----
    export_weights(m1, sc1, os.path.join(OUT, "weights_float_final", "model1"))
    export_weights(m2, sc2, os.path.join(OUT, "weights_float_final", "model2"))

    # ---- 导出 int8 ----
    X1s, X2s = sc1.transform(X1), sc2.transform(X2)
    export_weights(m1, sc1, os.path.join(OUT, "weights_int8_final", "model1"),
                   quantize=True, Xs=X1s)
    export_weights(m2, sc2, os.path.join(OUT, "weights_int8_final", "model2"),
                   quantize=True, Xs=X2s)

    # ---- 最终阈值(来自07的集成OOF, 真实交叉验证) ----
    oof = np.load(os.path.join(OUT, "oof_probs_ens.npy"))
    orig = np.load(os.path.join(CACHE, "feat_allaug_orig.npy"))
    y440 = np.load(os.path.join(CACHE, "feat_allaug_y.npy"))[orig]

    def table(cls_idx):
        rows = {}
        for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            trig = oof[:, cls_idx] >= t
            det = trig[y440 == cls_idx].mean()
            false = trig[y440 != cls_idx].mean()
            rows[round(t, 2)] = dict(det=round(float(det), 3),
                                     false=round(float(false), 3))
        return rows

    food_t, iso_t = table(1), table(2)
    print("最终集成模型阈值表(5折OOF, 440原始样本):")
    print("  阈值   饥饿检出/误触    焦虑检出/误触")
    for t in [0.4, 0.5, 0.6, 0.7]:
        print(f"  {t:.1f}    {food_t[t]['det']*100:5.1f}%/{food_t[t]['false']*100:4.1f}%"
              f"        {iso_t[t]['det']*100:5.1f}%/{iso_t[t]['false']*100:4.1f}%")

    acc = (oof.argmax(1) == y440).mean()
    meta = dict(model="2xMLP(512,256)概率平均集成",
                model1="295维+平衡权重+全类增强",
                model2="885维分段+平衡权重+food增强",
                cv_acc=round(float(acc), 4),
                per_class={c: round(float((oof[y440 == i].argmax(1) == i).mean()), 4)
                           for i, c in enumerate(CLASSES)},
                food_table=food_t, iso_table=iso_t,
                thresholds_recommended=dict(food=0.5, iso=0.6,
                                            note="连续2段确认出粮; 板上静音门限RMS<0.005"),
                quant_note="int8全量化损失约0.2pp, int16零损失, 均可用")
    with open(os.path.join(OUT, "final_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n集成模型CV准确率: {acc*100:.2f}%")
    print("已导出: weights_float_final/, weights_int8_final/, final_meta.json")


if __name__ == "__main__":
    main()
