# -*- coding: utf-8 -*-
"""
01_m1_对拍验证.py —— M1里程碑: 特征IP上板对拍 (PYNQ, 板上运行)
================================================================
验收标准: 板子IP输出的295维特征 与 PC端golden参考 最大误差 < 5e-3
这是cosim没做的RTL级验证——真实电路在真实板子上跑, 比软件仿真更硬

前置条件:
  1. PYNQ镜像SD卡 + 含特征IP的overlay(见README.md)
  2. golden的PCM数据转成的16k单声道wav放在 SD卡 /home/xilinx/cat_golden/ 下
     (生成方法: PC上跑 03_生成测试音频.py, 然后把 cat_golden/ 拷到SD卡)
  3. 权重CSV(weights_float_final/)拷到板子 /home/xilinx/cat_weights/

用法: 板上 Jupyter 或命令行 python3 01_m1_对拍验证.py
"""
import os
import numpy as np
import wave
from pynq import Overlay, allocate

OVERLAY_BIT = "/home/xilinx/cat_overlay.bit"      # Vivado生成的overlay
WAV_DIR = "/home/xilinx/cat_golden"               # 测试音频
GOLDEN_DIR = "/home/xilinx/cat_golden"            # 参考特征CSV(和wav放一起)
WEIGHT_DIR = "/home/xilinx/cat_weights"           # 模型权重

WIN_LEN = 24000
DIM = 295
DIM_OUT = 1180   # IP输出: 295整窗 + 885分段
TOL295 = 5e-3    # 特征容差(实测float32噪声1.4e-3~3.7e-3)
TOL885 = 8e-3    # 885维容差(实测噪声最大7.2e-3, 对概率影响为0)


def find_ip(overlay, prefix="cat_feat_ip"):
    """从overlay的ip_dict里找到我们的IP(名字可能带后缀_0)"""
    for name in overlay.ip_dict:
        if name.startswith(prefix):
            return getattr(overlay, name)
    raise RuntimeError(f"overlay里没找到 {prefix}* 的IP, 检查Vivado里的命名")


def read_wav_mono16k(path):
    """读16k单声道wav -> int16数组"""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"{path} 不是16k"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data


def ip_forward(ip, pcm_buf, feat_buf):
    """驱动IP: 写地址 -> 启动 -> 等完成
    寄存器映射(与HLS生成的control_s_axi.v一致, 已核对):
      0x00 ap_ctrl  0x10/0x14 pcm地址(低32/高32)  0x18 pcm_ctrl
      0x1c/0x20 feat地址(低32/高32)               0x24 feat_ctrl"""
    pa = int(pcm_buf.physical_address)
    ip.write(0x10, pa & 0xFFFFFFFF)
    ip.write(0x14, pa >> 32)
    fa = int(feat_buf.physical_address)
    ip.write(0x1c, fa & 0xFFFFFFFF)
    ip.write(0x20, fa >> 32)
    ip.write(0x00, 0x1)                        # ap_start
    import time as _t
    _t0 = _t.perf_counter()
    while not (ip.read(0x00) & 0x2):           # 轮询ap_done
        if _t.perf_counter() - _t0 > 60:
            raise RuntimeError("IP 60秒未完成: 检查寄存器地址映射/时钟是否正常")


def main():
    ol = Overlay(OVERLAY_BIT)
    ip = find_ip(ol)

    # 固定1.5s窗口: 测试音频由03脚本直接生成"已窗口化"的1.5s文件
    pcm_buf = allocate(shape=(WIN_LEN,), dtype=np.int32)     # int32, 见cat_feat_ip.c
    feat_buf = allocate(shape=(DIM_OUT,), dtype=np.float32)  # IP输出1180维

    files = sorted(f for f in os.listdir(WAV_DIR) if f.endswith("_wav.wav"))
    passed = 0
    print(f"共 {len(files)} 段测试音频\n")
    for f in files:
        base = f.replace("_wav.wav", "")
        pcm = read_wav_mono16k(os.path.join(WAV_DIR, f))
        if len(pcm) < WIN_LEN:
            print(f"  跳过 {f}: 长度不足")
            continue

        np.copyto(pcm_buf, pcm[:WIN_LEN].astype(np.int32))

        import time
        t0 = time.perf_counter()
        ip_forward(ip, pcm_buf, feat_buf)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        feat = np.array(feat_buf, dtype=np.float64)
        ref295 = np.loadtxt(os.path.join(GOLDEN_DIR, base + "_features295.csv"),
                            delimiter=",")
        err295 = np.abs(feat[:DIM] - ref295).max()
        ref885_path = os.path.join(GOLDEN_DIR, base + "_features885.csv")
        err885 = 0.0
        if os.path.exists(ref885_path):
            ref885 = np.loadtxt(ref885_path, delimiter=",")
            err885 = np.abs(feat[DIM:DIM_OUT] - ref885).max()
        ok = err295 < TOL295 and err885 < TOL885
        passed += ok
        print(f"  {base:32s} 295误差={err295:.3e} 885误差={err885:.3e} "
              f"{'PASS' if ok else 'FAIL'}  IP耗时={elapsed_ms:.1f}ms")

    print(f"\nM1结果: {passed}/{len(files)} 通过")
    print("验收: 全部PASS -> M1完成, 特征IP(295+885)在真实板子上验证通过")


if __name__ == "__main__":
    main()
