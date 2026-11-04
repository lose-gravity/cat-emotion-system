# -*- coding: utf-8 -*-
"""
06_模型对比.py —— 同一协议(GroupKFold先划分后增强)下的模型/特征对比
==================================================================
对比项:
  a. MLP(512,256)                基线(原结构)
  b. MLP(512,256)+类别平衡权重   针对亲昵类召回低
  c. MLP(1024,512,256)           更深更宽
  d. RandomForest(500)           树模型参考
  e. XGBoost                     梯度提升参考(可达上限)
  f. MLP(512,256)+分段特征       1.5s切成3段各提295维=885维, 保留时间结构
评估: 5折GroupKFold OOF准确率 + 各类召回 + 饥饿/焦虑检出-误触表
"""
import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_ref as fr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "_cache")
CLASSES = ["affection", "food", "isolation"]
SEED = 42


def build_segment_features():
    """885维分段特征: 固定1.5s切成3段各提295维。与02相同文件顺序+相同seed重放增强
    (groups一致保证GroupKFold划分一致; 增强噪声与02同分布, 不影响协议正确性)"""
    key = os.path.join(CACHE, "feat_seg_X.npy")
    if os.path.exists(key):
        return np.load(key)
    DATASET = os.path.join(os.path.dirname(OUT), "cat_dataset")
    np.random.seed(SEED)
    Xs, groups, ys, is_orig = [], [], [], []
    for gi, cls in enumerate(CLASSES):
        for f in sorted(os.listdir(os.path.join(DATASET, cls))):
            y = fr.load_wav(os.path.join(DATASET, cls, f))
            vers = fr.augment_food(y) if cls == "food" else [y]
            for v in vers:
                yw = fr.fixed_window(v)
                segs = [fr._extract_from_wave(yw[s * 8000:(s + 1) * 8000].astype(np.float64))
                        for s in range(3)]
                Xs.append(np.concatenate(segs))
                groups.append(f"{cls}/{f}")
                ys.append(gi)
                is_orig.append(v is y)
    Xs = np.array(Xs)
    np.save(key, Xs)
    np.save(os.path.join(CACHE, "feat_seg_groups.npy"), np.array(groups))
    np.save(os.path.join(CACHE, "feat_seg_y.npy"), np.array(ys))
    np.save(os.path.join(CACHE, "feat_seg_orig.npy"), np.array(is_orig))
    return Xs


def oof_eval(X, groups, ys, is_orig, make_model, needs_scaler=True, tag="",
             balance=False):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(ys), 3))
    for tr_idx, te_idx in gkf.split(X, ys, groups):
        te_idx = te_idx[is_orig[te_idx]]
        Xtr, Xte = X[tr_idx], X[te_idx]
        if needs_scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = make_model()
        if balance:
            w = {gi: 1.0 / (ys[tr_idx] == gi).sum() for gi in np.unique(ys)}
            sw = np.array([w[v] for v in ys[tr_idx]])
            model.fit(Xtr, ys[tr_idx], sample_weight=sw)
        else:
            model.fit(Xtr, ys[tr_idx])
        oof[te_idx] = model.predict_proba(Xte)
    acc = (oof.argmax(1) == ys)[is_orig].mean()
    line = f"  {tag:28s} CV={acc*100:5.2f}%"
    for gi, c in enumerate(CLASSES):
        m = (ys == gi) & is_orig
        line += f"  {c[:3]}={ (oof[m].argmax(1)==gi).mean()*100:5.1f}%"
    print(line)
    return oof


def main():
    X = np.load(os.path.join(CACHE, "feat_fixed_X.npy"))
    groups = np.load(os.path.join(CACHE, "feat_fixed_groups.npy"))
    ys = np.load(os.path.join(CACHE, "feat_fixed_y.npy"))
    is_orig = np.load(os.path.join(CACHE, "feat_fixed_orig.npy"))
    print(f"样本数: {len(X)} (含增强), 特征: {X.shape[1]}维\n")

    # a. 基线
    oof_a = oof_eval(X, groups, ys, is_orig,
                     lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                                           random_state=SEED, early_stopping=True,
                                           validation_fraction=0.15,
                                           learning_rate_init=0.0005), tag="a MLP(512,256)")
    # b. 类别平衡(sample_weight)
    oof_b = oof_eval(X, groups, ys, is_orig,
                     lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                                           random_state=SEED, early_stopping=True,
                                           validation_fraction=0.15, learning_rate_init=0.0005),
                     balance=True, tag="b MLP+平衡权重")
    # c. 更深
    oof_c = oof_eval(X, groups, ys, is_orig,
                     lambda: MLPClassifier(hidden_layer_sizes=(1024, 512, 256), max_iter=2000,
                                           random_state=SEED, early_stopping=True,
                                           validation_fraction=0.15, learning_rate_init=0.0005,
                                           alpha=1e-4), tag="c MLP(1024,512,256)")
    # d. 随机森林
    oof_d = oof_eval(X, groups, ys, is_orig,
                     lambda: RandomForestClassifier(n_estimators=500, max_depth=20,
                                                    random_state=SEED, n_jobs=-1,
                                                    class_weight="balanced"),
                     needs_scaler=False, tag="d RandomForest")
    # e. XGBoost
    try:
        import xgboost as xgb
        oof_e = oof_eval(X, groups, ys, is_orig,
                         lambda: xgb.XGBClassifier(n_estimators=400, max_depth=5,
                                                   learning_rate=0.05, subsample=0.9,
                                                   colsample_bytree=0.8, random_state=SEED,
                                                   eval_metric="mlogloss"),
                         needs_scaler=False, tag="e XGBoost")
        np.save(os.path.join(OUT, "oof_probs_xgb.npy"), oof_e)
    except ImportError:
        print("  e XGBoost 未安装, 跳过")

    # f. 分段特征 MLP
    Xs = build_segment_features()
    gs = np.load(os.path.join(CACHE, "feat_seg_groups.npy"))
    ys2 = np.load(os.path.join(CACHE, "feat_seg_y.npy"))
    os2 = np.load(os.path.join(CACHE, "feat_seg_orig.npy"))
    oof_f = oof_eval(Xs, gs, ys2, os2,
                     lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                                           random_state=SEED, early_stopping=True,
                                           validation_fraction=0.15, learning_rate_init=0.0005,
                                           alpha=1e-4), tag="f MLP+885维分段特征")

    np.save(os.path.join(OUT, "oof_probs_bal.npy"), oof_b)
    np.save(os.path.join(OUT, "oof_probs_rf.npy"), oof_d)

    # 阈值表对比(基线 vs 最佳候选)
    print("\n饥饿检出-误触对比(5折OOF):")
    print("  阈值  基线a      平衡b     RF d      XGB e")
    for t in [0.4, 0.5, 0.6, 0.7]:
        line = f"  {t:.1f}  "
        for oof in [oof_a, oof_b, oof_d, oof_e]:
            trig = oof[:, 1] >= t
            det = trig[(ys == 1) & is_orig].mean() * 100
            false = trig[(ys != 1) & is_orig].mean() * 100
            line += f" {det:5.1f}%/{false:4.1f}%"
        print(line)


if __name__ == "__main__":
    main()
