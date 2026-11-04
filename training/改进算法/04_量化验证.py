# -*- coding: utf-8 -*-
"""
04_量化验证.py —— int8对称量化: 精度损失评估 + 量化权重导出 + 阈值重校准
========================================================================
量化方案(per-tensor对称int8, HLS可直接实现):
  Wq = clip(round(W / w_scale), -127, 127)     w_scale = max|W|/127
  xq = clip(round(x / x_scale), -127, 127)     x_scale = max|x|/127
  h1f = relu(xq @ Wq1 + b1_f), h1q = clip(round(h1f/h1_scale),-127,127)
  (bias保持float32, HLS侧在int32累加域加bias, 见特征规格.md)
评估: 与02相同GroupKFold, OOF对比 float vs 量化 准确率, 并重算阈值表
"""
import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_ref as fr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "_cache")
CLASSES = ["affection", "food", "isolation"]
N_FOLDS = 5
SEED = 42


def quantize_weights(W, nbits):
    s = np.abs(W).max() / nbits
    Wq = np.clip(np.round(W / s), -nbits, nbits)
    return Wq.astype(np.int16), float(s)


def quant_forward(x_scaled, Ws, bs, scales, qbits, weight_only=False):
    """Ws: [W1q,W2q,W3q](int) bs:[b1,b2,b3] scales:(x_s,h1_s,h2_s,w1s,w2s,w3s)
    qbits: 127(int8)或32767(int16); weight_only=True时激活不量化"""
    x_s, h1_s, h2_s, w1s, w2s, w3s = scales
    if weight_only:
        h1 = np.maximum(x_scaled @ (Ws[0].astype(np.float64) * w1s) + bs[0], 0)
        h2 = np.maximum(h1 @ (Ws[1].astype(np.float64) * w2s) + bs[1], 0)
        logits = h2 @ (Ws[2].astype(np.float64) * w3s) + bs[2]
    else:
        xq = np.clip(np.round(x_scaled / x_s), -qbits, qbits) * x_s
        h1f = np.maximum(xq @ (Ws[0].astype(np.float64) * w1s) + bs[0], 0)
        h1q = np.clip(np.round(h1f / h1_s), -qbits, qbits) * h1_s
        h2f = np.maximum(h1q @ (Ws[1].astype(np.float64) * w2s) + bs[1], 0)
        h2q = np.clip(np.round(h2f / h2_s), -qbits, qbits) * h2_s
        logits = h2q @ (Ws[2].astype(np.float64) * w3s) + bs[2]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    X = np.load(os.path.join(CACHE, "feat_fixed_X.npy"))
    groups = np.load(os.path.join(CACHE, "feat_fixed_groups.npy"))
    ys = np.load(os.path.join(CACHE, "feat_fixed_y.npy"))
    is_orig = np.load(os.path.join(CACHE, "feat_fixed_orig.npy"))

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_f = np.zeros((len(ys), 3))       # float
    oof_q = {n: np.zeros((len(ys), 3)) for n in ["int8", "int16", "int8w"]}
    for tr_idx, te_idx in gkf.split(X, ys, groups):
        te_idx = te_idx[is_orig[te_idx]]
        scaler = StandardScaler().fit(X[tr_idx])
        model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                              random_state=SEED, early_stopping=True,
                              validation_fraction=0.15, learning_rate_init=0.0005)
        model.fit(scaler.transform(X[tr_idx]), ys[tr_idx])
        oof_f[te_idx] = model.predict_proba(scaler.transform(X[te_idx]))

        Xs = scaler.transform(X[tr_idx])
        h1 = np.maximum(Xs @ model.coefs_[0] + model.intercepts_[0], 0)
        h2 = np.maximum(h1 @ model.coefs_[1] + model.intercepts_[1], 0)
        for nbits, name in [(127, "int8"), (32767, "int16"), (127, "int8w")]:
            Wq, w_scales = [], []
            for W in model.coefs_:
                wq, s = quantize_weights(W, nbits)
                Wq.append(wq)
                w_scales.append(s)
            x_s = np.abs(Xs).max() / nbits
            h1_s = np.abs(h1).max() / nbits
            h2_s = np.abs(h2).max() / nbits
            scales = (x_s, h1_s, h2_s, *w_scales)
            weight_only = (name == "int8w")
            oof_q[name][te_idx] = quant_forward(scaler.transform(X[te_idx]),
                                                Wq, model.intercepts_, scales,
                                                nbits, weight_only=weight_only)

    acc_f = (oof_f.argmax(axis=1) == ys)[is_orig].mean()
    print(f"float      OOF准确率: {acc_f*100:.2f}%")
    for name in ["int8", "int16", "int8w"]:
        acc = (oof_q[name].argmax(axis=1) == ys)[is_orig].mean()
        print(f"{name:10s} OOF准确率: {acc*100:.2f}%   (损失 {(acc_f-acc)*100:.2f}pp)")

    print("\n饥饿检出-误触(5折OOF):")
    print("  阈值  int8       int16      int8w")
    for t in [0.4, 0.5, 0.6, 0.7]:
        line = f"  {t:.1f}  "
        for name in ["int8", "int16", "int8w"]:
            trig = oof_q[name][:, 1] >= t
            det = trig[(ys == 1) & is_orig].mean() * 100
            false = trig[(ys != 1) & is_orig].mean() * 100
            line += f" {det:5.1f}%/{false:4.1f}%"
        print(line)
    for name in ["int16", "int8w"]:
        np.save(os.path.join(OUT, f"oof_probs_{name}.npy"), oof_q[name])

    # ---- 最终模型: int16量化导出(DSP48原生精度) ----
    scaler = StandardScaler().fit(X)
    model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                          random_state=SEED, early_stopping=True,
                          validation_fraction=0.15, learning_rate_init=0.0005)
    model.fit(scaler.transform(X), ys)
    wdir = os.path.join(OUT, "weights_int16")
    os.makedirs(wdir, exist_ok=True)
    layer_scales = {}
    N = 32767
    for i, W in enumerate(model.coefs_):
        wq, s = quantize_weights(W, N)
        np.savetxt(os.path.join(wdir, f"W{i+1}_{wq.shape[1]}x{wq.shape[0]}.csv"),
                   wq.T, delimiter=",", fmt="%d")
        layer_scales[f"w{i+1}_scale"] = s
    for i, b in enumerate(model.intercepts_):
        np.savetxt(os.path.join(wdir, f"b{i+1}_{b.shape[0]}.csv"), b,
                   delimiter=",", fmt="%.9e")
    Xs = scaler.transform(X)
    h1 = np.maximum(Xs @ model.coefs_[0] + model.intercepts_[0], 0)
    h2 = np.maximum(h1 @ model.coefs_[1] + model.intercepts_[1], 0)
    layer_scales["x_scale"] = float(np.abs(Xs).max() / N)
    layer_scales["h1_scale"] = float(np.abs(h1).max() / N)
    layer_scales["h2_scale"] = float(np.abs(h2).max() / N)
    np.savetxt(os.path.join(wdir, "scaler_mean.csv"), scaler.mean_, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(wdir, "scaler_scale.csv"), scaler.scale_, delimiter=",", fmt="%.9e")
    with open(os.path.join(wdir, "layer_scales.json"), "w") as f:
        json.dump(layer_scales, f, indent=2)
    print(f"\nint16量化权重已导出: {wdir}")


if __name__ == "__main__":
    main()
