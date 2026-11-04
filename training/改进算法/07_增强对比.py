# -*- coding: utf-8 -*-
"""
07_增强对比.py —— 全类增强 + MLP集成(协议与02/06一致)
======================================================
增强方案: food 8倍(原样+±1.5半音+拉伸0.9/1.1+前4加噪)
          affection/isolation 3倍(原样+±1.5半音)
对比: MLP+平衡权重(295维) / MLP+平衡权重+分段特征(885维) / 两者概率平均集成
"""
import os
import sys
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
SEED = 42


def aug_all(y, cls):
    if cls == "food":
        return fr.augment_food(y)
    vers = [y]
    vers.append(librosa_shift(y, 1.5))
    vers.append(librosa_shift(y, -1.5))
    return vers


def librosa_shift(y, n):
    import librosa
    return librosa.effects.pitch_shift(y, sr=fr.SR, n_steps=n)


def build_allaug_cache():
    """全类增强特征缓存(固定窗口管线)"""
    key = os.path.join(CACHE, "feat_allaug_X.npy")
    if os.path.exists(key):
        return (np.load(key), np.load(os.path.join(CACHE, "feat_allaug_groups.npy")),
                np.load(os.path.join(CACHE, "feat_allaug_y.npy")),
                np.load(os.path.join(CACHE, "feat_allaug_orig.npy")))
    DATASET = os.path.join(os.path.dirname(OUT), "cat_dataset")
    np.random.seed(SEED)
    X, groups, ys, is_orig = [], [], [], []
    for gi, cls in enumerate(CLASSES):
        for f in sorted(os.listdir(os.path.join(DATASET, cls))):
            y = fr.load_wav(os.path.join(DATASET, cls, f))
            for v in aug_all(y, cls):
                X.append(fr.extract_fixed(v))
                groups.append(f"{cls}/{f}")
                ys.append(gi)
                is_orig.append(v is y)
    X, groups, ys, is_orig = map(np.array, (X, groups, ys, is_orig))
    for name, arr in [("feat_allaug_X", X), ("feat_allaug_groups", groups),
                      ("feat_allaug_y", ys), ("feat_allaug_orig", is_orig)]:
        np.save(os.path.join(CACHE, name + ".npy"), arr)
    return X, groups, ys, is_orig


def oof_mlp(X, groups, ys, is_orig, tag, balance=True):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(ys), 3))
    for tr_idx, te_idx in gkf.split(X, ys, groups):
        te_idx = te_idx[is_orig[te_idx]]
        sc = StandardScaler().fit(X[tr_idx])
        model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                              random_state=SEED, early_stopping=True,
                              validation_fraction=0.15, learning_rate_init=0.0005)
        w = {gi: 1.0 / (ys[tr_idx] == gi).sum() for gi in np.unique(ys)}
        sw = np.array([w[v] for v in ys[tr_idx]])
        model.fit(sc.transform(X[tr_idx]), ys[tr_idx], sample_weight=sw)
        oof[te_idx] = model.predict_proba(sc.transform(X[te_idx]))
    acc = (oof.argmax(1) == ys)[is_orig].mean()
    line = f"  {tag:26s} CV={acc*100:5.2f}%"
    for gi, c in enumerate(CLASSES):
        m = (ys == gi) & is_orig
        line += f"  {c[:3]}={(oof[m].argmax(1) == gi).mean()*100:5.1f}%"
    print(line)
    return oof


def main():
    X, groups, ys, is_orig = build_allaug_cache()
    print(f"全类增强样本数: {len(X)}\n")

    oof_295 = oof_mlp(X, groups, ys, is_orig, "MLP平衡+全类增强")

    # 分段特征(885维) + 全类增强
    Xs = np.load(os.path.join(CACHE, "feat_seg_X.npy"))
    gs = np.load(os.path.join(CACHE, "feat_seg_groups.npy"))
    ys2 = np.load(os.path.join(CACHE, "feat_seg_y.npy"))
    os2 = np.load(os.path.join(CACHE, "feat_seg_orig.npy"))
    oof_seg = oof_mlp(Xs, gs, ys2, os2, "MLP平衡+分段(仅food增强)")

    # 集成: 概率平均(在440原始样本上对齐: 两个缓存原始行顺序一致)
    oof_295o = oof_295[is_orig]
    oof_sego = oof_seg[os2]
    assert oof_295o.shape == oof_sego.shape == (440, 3)
    oof_ens = (oof_295o + oof_sego) / 2
    y440 = ys[is_orig]
    acc = (oof_ens.argmax(1) == y440).mean()
    line = f"  {'集成(概率平均)':26s} CV={acc*100:5.2f}%"
    for gi, c in enumerate(CLASSES):
        m = y440 == gi
        line += f"  {c[:3]}={(oof_ens[m].argmax(1) == gi).mean()*100:5.1f}%"
    print(line)
    np.save(os.path.join(OUT, "oof_probs_allaug.npy"), oof_295)
    np.save(os.path.join(OUT, "oof_probs_ens.npy"), oof_ens)

    print("\n饥饿检出-误触(5折OOF):")
    print("  阈值  全类增强     集成")
    for t in [0.4, 0.5, 0.6, 0.7]:
        line = f"  {t:.1f}  "
        for oof in [oof_295o, oof_ens]:
            trig = oof[:, 1] >= t
            det = trig[y440 == 1].mean() * 100
            false = trig[y440 != 1].mean() * 100
            line += f" {det:5.1f}%/{false:4.1f}%"
        print(line)


if __name__ == "__main__":
    main()
