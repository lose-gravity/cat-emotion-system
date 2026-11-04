# -*- coding: utf-8 -*-
# 用部署端(numpy)特征 + 模型, 对全部测试音频跑一遍, 评估部署端真实判断能力
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import deploy

sess, mean, scale = deploy.load_model()
base = "C:/Users/27907/Desktop/cat_dataset"
classes = ["affection", "food", "isolation"]
correct = 0
total = 0
per_cls_correct = {c: 0 for c in classes}
per_cls_total = {c: 0 for c in classes}
for idx, cls in enumerate(classes):
    folder = os.path.join(base, cls)
    for f in os.listdir(folder):
        if not f.lower().endswith('.wav'):
            continue
        wav = deploy.read_wav(os.path.join(folder, f))
        label, prob = deploy.predict(sess, mean, scale, wav)
        total += 1
        per_cls_total[cls] += 1
        if label == idx:
            correct += 1
            per_cls_correct[cls] += 1
print(f"部署端(numpy特征)对全部测试音频判断准确率: {correct/total*100:.1f}% ({correct}/{total})")
for c in classes:
    print(f"  {c}: {per_cls_correct[c]}/{per_cls_total[c]} = {per_cls_correct[c]/per_cls_total[c]*100:.1f}%")
