# -*- coding: utf-8 -*-
"""
mlp_numpy.py —— 板端MLP推理(纯numpy, 支持float与int8量化权重)
============================================================
推理公式(sklearn MLPClassifier, 295->512->256->3):
    x_scaled = (x - scaler_mean) / scaler_scale          # 标准化
    h1 = relu(x_scaled @ W1.T + b1)    W1: 512x295(CSV行主序)
    h2 = relu(h1 @ W2.T + b2)          W2: 256x512
    logits = h2 @ W3.T + b3            W3: 3x256
    prob = softmax(logits)                                # 3类概率
CSV权重布局(weights_float/): W{层}_{out}x{in}.csv 行主序(加载后转置再乘)
"""
import os
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class MLP:
    """前两层ReLU, 最后一层直出logits"""
    def __init__(self, weight_dir, quantized=False, layer_scales=None):
        """
        weight_dir: weights_float/ 或 weights_int8/ 目录
        quantized: 权重CSV是int8整数, 用layer_scales=(w1s,w2s,w3s,xs)反量化
        """
        def _find(prefix):
            hits = glob.glob(os.path.join(weight_dir, prefix + "_*.csv"))
            assert len(hits) == 1, f"{weight_dir} 下 {prefix}_*.csv 应恰好1个, 找到{len(hits)}个"
            return np.loadtxt(hits[0], delimiter=",")

        W1 = _find("W1")
        b1 = _find("b1")
        W2 = _find("W2")
        b2 = _find("b2")
        W3 = _find("W3")
        b3 = _find("b3")
        self.mean = np.loadtxt(os.path.join(weight_dir, "scaler_mean.csv"), delimiter=",")
        self.scale = np.loadtxt(os.path.join(weight_dir, "scaler_scale.csv"), delimiter=",")
        if quantized:
            w1s, w2s, w3s, xs = layer_scales
            self.W1, self.W2, self.W3 = W1 * w1s, W2 * w2s, W3 * w3s
            self.x_scale = xs
        else:
            self.W1, self.W2, self.W3 = W1, W2, W3
            self.x_scale = None
        self.b1, self.b2, self.b3 = b1, b2, b3

    def forward(self, x):
        if self.x_scale is not None:
            x = np.round((x - self.mean) / self.scale / self.x_scale) * self.x_scale
        else:
            x = (x - self.mean) / self.scale
        h1 = np.maximum(x @ self.W1.T + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2.T + self.b2, 0.0)
        logits = h2 @ self.W3.T + self.b3
        return logits, softmax(logits)


def load_final_model(quantized=False, layer_scales=None):
    sub = "weights_int8" if quantized else "weights_float"
    return MLP(os.path.join(_HERE, sub), quantized=quantized,
               layer_scales=layer_scales)


def load_ensemble(quantized=False, scales=None):
    """最终集成模型: (模型1(295维), 模型2(885维)); 推理=两模型概率平均"""
    base = "weights_int8_final" if quantized else "weights_float_final"
    s1 = s2 = None
    if quantized:
        import json as _json
        s1 = _read_scales(os.path.join(_HERE, base, "model1", "layer_scales.json"))
        s2 = _read_scales(os.path.join(_HERE, base, "model2", "layer_scales.json"))
    m1 = MLP(os.path.join(_HERE, base, "model1"), quantized=quantized, layer_scales=s1)
    m2 = MLP(os.path.join(_HERE, base, "model2"), quantized=quantized, layer_scales=s2)
    return m1, m2


def _read_scales(path):
    import json as _json
    with open(path) as f:
        sj = _json.load(f)
    return (sj["w1_scale"], sj["w2_scale"], sj["w3_scale"], sj["x_scale"])


def predict_ensemble(feat295, feat885, m1=None, m2=None):
    """295维+885维特征 -> (logits, prob)"""
    if m1 is None or m2 is None:
        m1, m2 = load_ensemble()
    _, p1 = m1.forward(feat295)
    _, p2 = m2.forward(feat885)
    p = (p1 + p2) / 2
    return p.argmax(), p


if __name__ == "__main__":
    m = load_final_model()
    rng = np.random.default_rng(0)
    x = rng.normal(size=295)
    logits, prob = m.forward(x)
    print("冒烟: logits =", np.round(logits, 3), " prob =", np.round(prob, 3), " 和=", prob.sum())
