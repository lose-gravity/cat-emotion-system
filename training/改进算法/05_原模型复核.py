# -*- coding: utf-8 -*-
"""
05_原模型复核.py —— 用本机librosa 1.0.0复现原模型的90.2%声明
============================================================
背景: 原模型在另一台电脑训练(原版trim管线+未知librosa版本)。
本脚本在本机用 librosa 1.0.0 重跑原版管线 + 原ONNX模型 + 原scaler,
对全量440段评估, 验证:
  1. 原模型文件完好、输出格式与deploy.py解析一致
  2. librosa 1.0.0 与原训练版本的特征行为是否等效(90.2%能否复现)
"""
import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_ref as fr

OUT = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(os.path.dirname(OUT), "02-部署应用")
DATASET = os.path.join(os.path.dirname(OUT), "cat_dataset")
CLASSES = ["affection", "food", "isolation"]


def main():
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(DEPLOY, "cat_emotion.onnx"),
                                providers=["CPUExecutionProvider"])
    with open(os.path.join(DEPLOY, "scaler.json"), encoding="utf-8") as f:
        sc = json.load(f)
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)

    correct = 0
    per = {c: [0, 0] for c in CLASSES}
    for gi, cls in enumerate(CLASSES):
        for f in sorted(os.listdir(os.path.join(DATASET, cls))):
            y = fr.load_wav(os.path.join(DATASET, cls, f))
            feat = fr.extract_legacy(y).astype(np.float32)
            feat = (feat - mean) / scale
            out = sess.run(None, {sess.get_inputs()[0].name: feat.reshape(1, -1)})
            label = int(out[0][0])
            per[cls][1] += 1
            if label == gi:
                per[cls][0] += 1
                correct += 1
    total = sum(v[1] for v in per.values())
    print(f"本机复现(原ONNX + librosa 1.0.0 原版管线): {correct}/{total} = {correct/total*100:.2f}%")
    for c in CLASSES:
        print(f"  {c}: {per[c][0]}/{per[c][1]} = {per[c][0]/per[c][1]*100:.1f}%")
    print("\n原队友声明: 90.2% (亲昵81% / 饥饿93.5% / 焦虑94.1%)")


if __name__ == "__main__":
    main()
