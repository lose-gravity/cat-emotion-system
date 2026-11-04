# -*- coding: utf-8 -*-
"""
08_双检测器.py —— 面向产品触发逻辑的二分类器方案
================================================
思路: 产品只需要两个触发器(出粮/安抚), 不关心三分类argmax。
训练两个独立二分类器: 饥饿检测器(food vs 其他), 焦虑检测器(iso vs 其他),
决策逻辑: food分>=Tf → 出粮; 否则 iso分>=Ti → 安抚; 否则不动作。
对比: 三分类MLP的概率阈值法(基线) vs 专用二分类器。
协议与02一致(GroupKFold, 先划分后增强, OOF仅原始440段)。
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "_cache")
CLASSES = ["affection", "food", "isolation"]
SEED = 42
N_FOLDS = 5


def oof_binary(X, groups, ys, is_orig, target_cls, make_model, tag):
    """target_cls: 该类别=1, 其余=0。返回OOF的类1概率"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(ys))
    ybin = (ys == target_cls).astype(int)
    for tr_idx, te_idx in gkf.split(X, ys, groups):
        te_idx = te_idx[is_orig[te_idx]]
        sc = StandardScaler().fit(X[tr_idx])
        model = make_model()
        model.fit(sc.transform(X[tr_idx]), ybin[tr_idx])
        oof[te_idx] = model.predict_proba(sc.transform(X[te_idx]))[:, 1]
    det = oof[(ys == target_cls) & is_orig]
    other = oof[(ys != target_cls) & is_orig]
    for t in [0.4, 0.5, 0.6, 0.7]:
        d = (det >= t).mean() * 100
        f = (other >= t).mean() * 100
        print(f"    {tag:14s} t={t:.1f}  检出{d:5.1f}%  误触{f:5.1f}%")
    return oof


def main():
    X = np.load(os.path.join(CACHE, "feat_allaug_X.npy"))
    groups = np.load(os.path.join(CACHE, "feat_allaug_groups.npy"))
    ys = np.load(os.path.join(CACHE, "feat_allaug_y.npy"))
    is_orig = np.load(os.path.join(CACHE, "feat_allaug_orig.npy"))

    print("== 饥饿检测器(food vs 其他) ==")
    oof_food_mlp = oof_binary(X, groups, ys, is_orig, 1,
                              lambda: MLPClassifier(hidden_layer_sizes=(512, 256),
                                                    max_iter=2000, random_state=SEED,
                                                    early_stopping=True,
                                                    validation_fraction=0.15,
                                                    learning_rate_init=0.0005),
                              "二分类MLP")
    oof_food_lr = oof_binary(X, groups, ys, is_orig, 1,
                             lambda: LogisticRegression(C=1.0, max_iter=2000),
                             "二分类LogReg")

    print("\n== 焦虑检测器(iso vs 其他) ==")
    oof_iso_mlp = oof_binary(X, groups, ys, is_orig, 2,
                             lambda: MLPClassifier(hidden_layer_sizes=(512, 256),
                                                   max_iter=2000, random_state=SEED,
                                                   early_stopping=True,
                                                   validation_fraction=0.15,
                                                   learning_rate_init=0.0005),
                             "二分类MLP")
    oof_iso_lr = oof_binary(X, groups, ys, is_orig, 2,
                            lambda: LogisticRegression(C=1.0, max_iter=2000),
                            "二分类LogReg")

    # 产品决策逻辑评估: food分>=Tf出粮, 否则iso分>=Ti安抚, 否则不动作
    print("\n== 产品决策准确率(二分类MLP, OOF原始440段) ==")
    y440 = ys[is_orig]
    for Tf in [0.5, 0.6]:
        for Ti in [0.5, 0.6]:
            food = oof_food_mlp[is_orig] >= Tf
            iso = (~food) & (oof_iso_mlp[is_orig] >= Ti)
            pred = np.where(food, 1, np.where(iso, 2, 0))
            acc = (pred == y440).mean() * 100
            # 各类行为正确率
            r = []
            for gi in range(3):
                r.append((pred[y440 == gi] == gi).mean() * 100)
            print(f"  Tf={Tf} Ti={Ti}: 决策准确率{acc:5.1f}%  "
                  f"(亲昵正确{r[0]:5.1f}% 饥饿正确{r[1]:5.1f}% 焦虑正确{r[2]:5.1f}%)")

    np.save(os.path.join(OUT, "oof_food_bin.npy"), oof_food_mlp)
    np.save(os.path.join(OUT, "oof_iso_bin.npy"), oof_iso_mlp)


if __name__ == "__main__":
    main()
