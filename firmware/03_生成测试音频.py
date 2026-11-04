# -*- coding: utf-8 -*-
"""
03_生成测试音频.py —— PC端: 把golden的PCM转成板上可用的16k单声道wav
================================================================
产物: cat_golden/ 目录(拷到SD卡 /home/xilinx/cat_golden/):
  - 40段 *_pcm.csv -> *_wav.wav (16kHz, 16bit, 单声道, 长度=24000点=1.5s)
  - 40段 *_features295.csv (参考特征, 01脚本对拍用)
  - soothing.wav (安抚音频占位: 1s静音, M3时换成真安抚音频)
用法: PC上 py -3.12 03_生成测试音频.py
"""
import os
import wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(os.path.dirname(HERE), "04-改进算法", "golden")
OUT = os.path.join(HERE, "cat_golden")
os.makedirs(OUT, exist_ok=True)


def main():
    n = 0
    for f in sorted(os.listdir(GOLDEN)):
        if not f.endswith("_pcm.csv"):
            continue
        base = f.replace("_pcm.csv", "")
        pcm = np.loadtxt(os.path.join(GOLDEN, f), dtype=np.int16)

        # wav输出(名字带_wav区分, 01脚本按此找)
        with wave.open(os.path.join(OUT, base + "_wav.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())

        # 参考特征直接拷过去(295+885)
        import shutil
        shutil.copy(os.path.join(GOLDEN, base + "_features295.csv"), OUT)
        shutil.copy(os.path.join(GOLDEN, base + "_features885.csv"), OUT)
        n += 1

    # 安抚音频占位(1s静音, M3时替换成录好的安抚话术)
    with wave.open(os.path.join(OUT, "soothing.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    print(f"生成完成: {n} 段测试音频 + 40 份参考特征 -> {OUT}")
    print(f"把这个目录整个拷到SD卡 /home/xilinx/cat_golden/ 即可")


if __name__ == "__main__":
    main()
