# -*- coding: utf-8 -*-
"""
10_导出中间量.py —— HLS调试用逐阶段参考值
==========================================
对golden/里40段音频, 每段导出特征链路的中间量(纯numpy版, float64):
  mel_db_151x128.csv   功率谱->mel滤波->log(db, ref=1, top_db=80)  (151帧x128)
  mfcc_40x151.csv      DCT-II(ortho)取前40
  d1_40x151.csv        savgol delta一阶
  d2_40x151.csv        savgol delta二阶
  centroid_47.csv      谱质心(47帧)
  rms_47.csv           RMS(47帧)
  zcr_47.csv           过零率(47帧)
输出: golden/interm/{文件名}/ 下, 与features295.csv最终值构成完整对拍链
HLS调试顺序建议: 喂pcm -> 对mel_db -> 对mfcc -> 对d1/d2 -> 对质心/rms/zcr -> 对295统计值
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_numpy as fn

OUT = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(OUT, "golden")
INTERM = os.path.join(GOLDEN, "interm")
os.makedirs(INTERM, exist_ok=True)


def dump_one(base):
    d = os.path.join(INTERM, base)
    os.makedirs(d, exist_ok=True)
    pcm = np.loadtxt(os.path.join(GOLDEN, base + "_pcm.csv"))
    y = pcm.astype(np.float64) / 32768.0

    m = fn.load_matrices()
    S = np.abs(fn._stft(y, fn.N_FFT, fn.HOP, fn._window_centered(m["hann400"], fn.N_FFT))) ** 2
    mel_spec = S @ m["mel"].T
    mel_db = fn._power_to_db(mel_spec)
    mfcc = m["dct"] @ mel_db.T
    d1 = mfcc @ m["sg1"][151].T
    d2 = mfcc @ m["sg2"][151].T
    cent = fn._centroid(y)
    rms = fn._rms(y)
    zcr = fn._zcr(y)

    np.savetxt(os.path.join(d, "mel_db_151x128.csv"), mel_db, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "mfcc_40x151.csv"), mfcc, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "d1_40x151.csv"), d1, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "d2_40x151.csv"), d2, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "centroid_47.csv"), cent, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "rms_47.csv"), rms, delimiter=",", fmt="%.9e")
    np.savetxt(os.path.join(d, "zcr_47.csv"), zcr, delimiter=",", fmt="%.9e")


def main():
    bases = sorted({f.rsplit("_", 1)[0] for f in os.listdir(GOLDEN)
                    if f.endswith("_pcm.csv")})
    for b in bases:
        dump_one(b)
    print(f"中间量导出完成: {len(bases)}段 -> golden/interm/")
    print("每段含: mel_db(151x128) mfcc(40x151) d1/d2(40x151) centroid/rms/zcr(47)")


if __name__ == "__main__":
    main()
