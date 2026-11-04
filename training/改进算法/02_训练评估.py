# -*- coding: utf-8 -*-
"""
02_训练评估.py —— 修正版训练与评估(修复数据泄露, 固定窗口, 真实指标)
====================================================================
与原版的关键差异:
  1. 先按"源文件"分组划分(GroupKFold 5折), 再对训练折做饥饿类增强
     —— 修复原版"先增强再划分"导致增强变体跨训练/测试集的数据泄露
  2. 固定1.5s窗口(去静音裁剪), 与板端/HLS实现一致
  3. 阈值权衡表由OOF(袋外)概率真实计算, 非硬编码文本
  4. 全零帧谱质心定义为0(原版librosa会产生nan)

产物(04-改进算法/):
  weights_float/W*.csv, b*.csv, scaler_mean.csv, scaler_scale.csv  —— HLS权重交付
  model_meta.json      —— 模型结构/类顺序/阈值建议/CV指标
  _cache/features_*.npy —— 特征缓存(重跑秒级)
  oof_probs.npy, oof_labels.npy —— 袋外概率(阈值校准/量化验证用)
"""
import os
import sys
import json
import time
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
os.makedirs(CACHE, exist_ok=True)
DATASET = os.path.join(os.path.dirname(OUT), "cat_dataset")
CLASSES = ["affection", "food", "isolation"]
N_FOLDS = 5
SEED = 42
np.random.seed(SEED)


# ================= 1. 特征提取(带缓存) =================
def cache_path(name):
    return os.path.join(CACHE, name + ".npy")


def get_features(pipeline):
    """返回 X(所有样本特征), groups(源文件idx), class_of_sample, is_orig(是否原始样本)"""
    key = "feat_" + pipeline
    if os.path.exists(cache_path(key + "_X")):
        X = np.load(cache_path(key + "_X"))
        groups = np.load(cache_path(key + "_groups"))
        ys = np.load(cache_path(key + "_y"))
        is_orig = np.load(cache_path(key + "_orig"))
        return X, groups, ys, is_orig

    t0 = time.time()
    X, groups, ys, is_orig = [], [], [], []
    extract = fr.extract_fixed if pipeline == "fixed" else fr.extract_legacy
    for gi, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET, cls)
        files = sorted(os.listdir(folder))
        for fi, f in enumerate(files):
            y = fr.load_wav(os.path.join(folder, f))
            if cls == "food":
                for v in fr.augment_food(y):
                    X.append(extract(v))
                    groups.append(f"{cls}/{f}")
                    ys.append(gi)
                    is_orig.append(v is y)
            else:
                X.append(extract(y))
                groups.append(f"{cls}/{f}")
                ys.append(gi)
                is_orig.append(True)
            print(f"  [{pipeline}] {cls} {fi+1}/{len(files)}  t={time.time()-t0:.0f}s", end="\r")
    X = np.array(X)
    groups = np.array(groups)
    ys = np.array(ys)
    is_orig = np.array(is_orig)
    np.save(cache_path(key + "_X"), X)
    np.save(cache_path(key + "_groups"), groups)
    np.save(cache_path(key + "_y"), ys)
    np.save(cache_path(key + "_orig"), is_orig)
    print(f"\n  [{pipeline}] 特征完成: {X.shape}, 用时 {time.time()-t0:.0f}s")
    return X, groups, ys, is_orig


# ================= 2. 交叉验证 =================
def crossval(X, groups, ys, is_orig, tag):
    """GroupKFold: 训练折=原始样本+饥饿增强; 测试折=仅原始样本(全440段各出现一次)"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_prob = np.zeros((len(ys), len(CLASSES)))
    oof_label = np.zeros(len(ys), dtype=int)
    accs, recalls = [], {c: [] for c in CLASSES}

    for k, (tr_idx, te_idx) in enumerate(gkf.split(X, ys, groups)):
        # 测试折只用原始样本
        te_idx = te_idx[is_orig[te_idx]]
        scaler = StandardScaler().fit(X[tr_idx])
        model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                              random_state=SEED, early_stopping=True,
                              validation_fraction=0.15, learning_rate_init=0.0005)
        model.fit(scaler.transform(X[tr_idx]), ys[tr_idx])
        prob = model.predict_proba(scaler.transform(X[te_idx]))
        pred = prob.argmax(axis=1)
        acc = (pred == ys[te_idx]).mean()
        accs.append(acc)
        for gi, cls in enumerate(CLASSES):
            m = ys[te_idx] == gi
            if m.sum():
                recalls[cls].append((pred[m] == gi).mean())
        oof_prob[te_idx] = prob
        oof_label[te_idx] = ys[te_idx]
        print(f"  fold{k+1} acc={acc:.3f}")

    print(f"[{tag}] CV准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    for cls in CLASSES:
        print(f"    {cls}: 召回 {np.mean(recalls[cls])*100:.1f}%")
    return oof_prob, oof_label


def threshold_table(oof_prob, oof_label):
    """真实计算阈值权衡表(仅原始样本)"""
    print("\n饥饿类阈值权衡(5折OOF):")
    print("  阈值   检出率   误触率")
    table = {}
    for t in np.arange(0.40, 0.96, 0.05):
        trig = oof_prob[:, 1] >= t
        det = trig[oof_label == 1].mean()
        false = trig[oof_label != 1].mean()
        table[round(float(t), 2)] = dict(det=round(float(det), 3), false=round(float(false), 3))
        print(f"  {t:.2f}   {det*100:5.1f}%   {false*100:5.1f}%")
    print("\n焦虑类阈值权衡(5折OOF):")
    print("  阈值   检出率   误触率")
    table_iso = {}
    for t in np.arange(0.40, 0.96, 0.05):
        trig = oof_prob[:, 2] >= t
        det = trig[oof_label == 2].mean()
        false = trig[oof_label != 2].mean()
        table_iso[round(float(t), 2)] = dict(det=round(float(det), 3), false=round(float(false), 3))
        print(f"  {t:.2f}   {det*100:5.1f}%   {false*100:5.1f}%")
    return table, table_iso


# ================= 3. 最终模型训练与导出 =================
def train_final_export(X, ys):
    scaler = StandardScaler().fit(X)
    model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=2000,
                          random_state=SEED, early_stopping=True,
                          validation_fraction=0.15, learning_rate_init=0.0005)
    model.fit(scaler.transform(X), ys)
    wdir = os.path.join(OUT, "weights_float")
    os.makedirs(wdir, exist_ok=True)
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        np.savetxt(os.path.join(wdir, f"W{i+1}_{W.shape[1]}x{W.shape[0]}.csv"), W.T,
                   delimiter=",", fmt="%.9e")
        np.savetxt(os.path.join(wdir, f"b{i+1}_{b.shape[0]}.csv"), b, delimiter=",",
                   fmt="%.9e")
        print(f"W{i+1}: {W.shape[0]}x{W.shape[1]} (转置为 {W.shape[1]}x{W.shape[0]} 行主序)  b{i+1}: {b.shape[0]}")
    np.savetxt(os.path.join(wdir, "scaler_mean.csv"), scaler.mean_, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(wdir, "scaler_scale.csv"), scaler.scale_, delimiter=",", fmt="%.9e")
    return model, scaler


def main():
    # ---- 管线A: 固定窗口(部署规格) ----
    print("==== 固定窗口管线(部署规格) ====")
    Xf, gf, yf, of = get_features("fixed")
    oof_prob_f, oof_label_f = crossval(Xf, gf, yf, of, "fixed")

    # ---- 管线B: 原版trim(对比基准, 正确划分后) ----
    print("\n==== 原版trim管线(对比基准) ====")
    Xl, gl, yl, ol = get_features("legacy")
    oof_prob_l, oof_label_l = crossval(Xl, gl, yl, ol, "legacy")

    # ---- 阈值权衡表(部署规格管线) ----
    table, table_iso = threshold_table(oof_prob_f, oof_label_f)

    # ---- 最终模型: 固定窗口管线全量训练 + 导出 ----
    print("\n==== 最终模型(固定窗口, 全量训练) ====")
    model, scaler = train_final_export(Xf, yf)

    # 全量数据上的"训练集"准确率(仅参考, 不代表泛化)
    prob_all = model.predict_proba(scaler.transform(Xf))
    acc_all = (prob_all.argmax(axis=1) == yf).mean()
    print(f"全量数据(含增强)自评准确率: {acc_all*100:.2f}% (仅参考)")

    np.save(os.path.join(OUT, "oof_probs.npy"), oof_prob_f)
    np.save(os.path.join(OUT, "oof_labels.npy"), oof_label_f)

    meta = dict(
        pipeline="fixed_1.5s", sr=fr.SR, n_mfcc=fr.N_MFCC, win=fr.WIN, hop=fr.HOP,
        win_samples=fr.FIXED_LEN, feature_dim=int(Xf.shape[1]),
        classes=CLASSES, hidden=(512, 256), activation="relu",
        cv_acc_mean=float((oof_prob_f.argmax(axis=1) == oof_label_f)[of].mean()),
        legacy_cv_mean=float((oof_prob_l.argmax(axis=1) == oof_label_l)[ol].mean()),
        food_table=table, iso_table=table_iso,
        librosa_version="1.0.0", numpy_version=np.__version__,
        note="GroupKFold先划分后增强; OOF仅含440原始样本; 全零帧质心=0",
    )
    with open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n已导出: weights_float/, model_meta.json, oof_probs.npy")


if __name__ == "__main__":
    main()
