# -*- coding: utf-8 -*-
"""
features_numpy.py —— 纯numpy特征提取器(板端PS可跑, HLS的参考实现)
================================================================
与 librosa 1.0.0 在"固定1.5s窗口"输入下逐位一致(误差<1e-3, 见对拍脚本)。
依赖: 仅 numpy + 00_生成矩阵.py 导出的CSV(或运行时用librosa生成, 见load_matrices)
不依赖: librosa/scipy/soundfile(板端不需要安装)

与训练端 librosa 版的唯一有意差异:
  - 全零帧(静音补零区域)的谱质心: librosa 会算出 nan, 本实现定义为 0
    (nan会让MLP输出垃圾, 真实麦克风场景必然有全零帧, 训练端已做同样处理)
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ===== 固定参数(与训练一致, 禁止改动) =====
SR = 16000
N_FFT = 2048
WIN = 400       # 25ms
HOP = 160       # 10ms
N_MELS = 128
N_MFCC = 40
WIN_LEN = 24000          # 固定窗口 1.5s
BLOCK = 1600             # 能量窗选择步进 0.1s
NWIN_BLOCKS = 15         # 1.5s = 15块
HOP_RMS = 512            # rms/zcr/centroid 的帧移(与mfcc的160不同!)

_matrices = None


def load_matrices(use_csv=True):
    """加载mel/DCT/窗/savgol矩阵。use_csv=False时直接用librosa生成(仅PC端用)"""
    global _matrices
    if _matrices is not None:
        return _matrices
    if use_csv and os.path.exists(os.path.join(_HERE, "mel_128x1025.csv")):
        mel = np.loadtxt(os.path.join(_HERE, "mel_128x1025.csv"), delimiter=",")
        dct = np.loadtxt(os.path.join(_HERE, "dct_40x128.csv"), delimiter=",")
        hann400 = np.loadtxt(os.path.join(_HERE, "hann_400.csv"), delimiter=",")
        hann2048 = np.loadtxt(os.path.join(_HERE, "hann_2048.csv"), delimiter=",")
        sg1 = {151: np.loadtxt(os.path.join(_HERE, "savgol_d1.csv"), delimiter=","),
               51: np.loadtxt(os.path.join(_HERE, "savgol_d1_51.csv"), delimiter=",")}
        sg2 = {151: np.loadtxt(os.path.join(_HERE, "savgol_d2.csv"), delimiter=","),
               51: np.loadtxt(os.path.join(_HERE, "savgol_d2_51.csv"), delimiter=",")}
    else:
        import librosa
        mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0,
                                  fmax=SR / 2, htk=False, norm="slaney")
        dct = np.zeros((N_MFCC, N_MELS))
        for k in range(N_MFCC):
            dct[k] = np.cos(np.pi * k * (2 * np.arange(N_MELS) + 1) / (2 * N_MELS))
        dct *= np.sqrt(2.0 / N_MELS)
        dct[0] /= np.sqrt(2.0)
        import scipy.signal
        hann400 = scipy.signal.get_window("hann", WIN, fftbins=True)
        hann2048 = scipy.signal.get_window("hann", N_FFT, fftbins=True)
        sg1 = sg2 = None  # 运行时生成见 _delta
        raise NotImplementedError("savgol运行时生成请先用00_生成矩阵.py导出CSV")
    _matrices = dict(mel=mel, dct=dct, hann400=hann400, hann2048=hann2048,
                     sg1=sg1, sg2=sg2)
    return _matrices


# ===== 基础算子(与librosa等价实现) =====
def _frame(y, frame_length, hop_length):
    """y(1D) -> (n_frames, frame_length), 同 librosa.util.frame"""
    n = y.shape[0]
    if n < frame_length:
        return np.zeros((0, frame_length))
    n_frames = 1 + (n - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def _window_centered(win, n_fft):
    """把窗居中嵌入n_fft长度(同librosa pad_center)"""
    out = np.zeros(n_fft)
    left = (n_fft - len(win)) // 2
    out[left:left + len(win)] = win
    return out


def _stft(y, n_fft, hop, window_full):
    """经典实现: 两侧补n_fft//2零 -> 分帧加窗 -> rfft。与librosa 1.0优化版帧内容一致"""
    ypad = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")
    frames = _frame(ypad, n_fft, hop).astype(np.float64)
    frames = frames * window_full
    return np.fft.rfft(frames, n=n_fft, axis=1)  # (n_frames, 1025)


def _power_to_db(S):
    """同 librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=80)"""
    db = 10.0 * np.log10(np.maximum(S, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    return db


def _dct_ortho(S_db):
    """DCT-II ortho -> (40, n_frames)"""
    return load_matrices()["dct"] @ S_db.T


def _mfcc(y):
    m = load_matrices()
    S = np.abs(_stft(y, N_FFT, HOP, _window_centered(m["hann400"], N_FFT))) ** 2
    mel_spec = S @ m["mel"].T            # (frames, 128)
    return _dct_ortho(_power_to_db(mel_spec))


def _rms(y):
    ypad = np.pad(y, (1024, 1024), mode="constant")
    x = _frame(ypad, 2048, HOP_RMS)
    return np.sqrt(np.mean(x ** 2, axis=1))


def _zcr(y):
    ypad = np.pad(y, (1024, 1024), mode="edge")   # 注意: 此处是edge不是constant
    x = _frame(ypad, 2048, HOP_RMS)
    # librosa zero_crossings: |y|<=1e-10 的样本先归零(不计过零), 再数符号翻转
    x = np.where(np.abs(x) <= 1e-10, 0.0, x)
    sign = np.signbit(x)
    crossings = np.abs(np.diff(sign, axis=1)).sum(axis=1)
    return crossings / 2048.0


def _centroid(y):
    m = load_matrices()
    S = np.abs(_stft(y, N_FFT, HOP_RMS, m["hann2048"]))    # 幅度谱(power=1)
    Ssum = S.sum(axis=1, keepdims=True)
    S_norm = np.where(Ssum > 0, S / Ssum, 0.0)             # 全零帧->0 (librosa此处会nan)
    freq = SR * np.arange(N_FFT // 2 + 1) / N_FFT
    return (S_norm * freq).sum(axis=1)


def _delta(mfcc, order):
    """librosa 1.0 delta = savgol_filter(width=9, polyorder=order, deriv=order, mode='interp')
    用预生成的线性算子矩阵实现(151帧: 1.5s窗口; 51帧: 0.5s切片)"""
    T = load_matrices()["sg1" if order == 1 else "sg2"]
    n = mfcc.shape[1]
    if n not in T:
        raise ValueError(f"savgol矩阵只有{T.keys()}帧版本, 当前{n}帧, "
                         f"请用00_生成矩阵.py重新生成")
    return mfcc @ T[n].T


def _percentile_linear(x, p):
    """numpy percentile method='linear' 的等价实现"""
    x = np.sort(x)
    n = len(x)
    idx = (n - 1) * p / 100.0
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    w = idx - lo
    return x[lo] * (1.0 - w) + x[hi] * w


def _stats(x):
    return np.concatenate([[_percentile_linear(x, p) for p in (10, 50, 90)],
                           [x.mean(), x.std()]])


# ===== 固定窗口 =====
def fixed_window(y, win_len=WIN_LEN):
    """固定窗口: 不足补零(尾部); 超长取0.1s块能量和最大的连续窗口"""
    n = len(y)
    if n < win_len:
        return np.pad(y, (0, win_len - n))
    n_win_blocks = win_len // BLOCK
    n_blocks = n // BLOCK
    if n_blocks < n_win_blocks:
        return np.pad(y[:n_win_blocks * BLOCK], (0, max(0, win_len - n)))
    e = np.array([np.sum(y[i * BLOCK:(i + 1) * BLOCK] ** 2) for i in range(n_blocks)])
    csum = np.convolve(e, np.ones(n_win_blocks), mode="valid")
    start = int(np.argmax(csum)) * BLOCK
    return y[start:start + win_len]


# ===== 主入口: 波形 -> 295维特征 =====
def extract_features_from_wave(y, win_len=WIN_LEN):
    """y: 16kHz float波形(任意长度, 内部先做固定窗口) -> 295维特征
    win_len=24000: 1.5s窗口(151帧); win_len=8000: 0.5s切片(51帧)"""
    y = fixed_window(y, win_len).astype(np.float64)
    mfcc = _mfcc(y)                       # (40, 151或51)
    d1 = _delta(mfcc, 1)
    d2 = _delta(mfcc, 2)
    cent = _centroid(y)                   # (47或16,)
    rms = _rms(y)
    zcr = _zcr(y)
    per_mfcc = np.concatenate([_stats(mfcc[i]) for i in range(N_MFCC)])
    return np.concatenate([per_mfcc, d1.mean(axis=1), d2.mean(axis=1),
                           _stats(cent), _stats(rms), _stats(zcr)])


def extract_features_segmental(y):
    """1.5s固定窗口切成3段0.5s各提295维 -> 885维分段特征(集成模型2用)"""
    y = fixed_window(y, WIN_LEN).astype(np.float64)
    return np.concatenate([extract_features_from_wave(y[s * 8000:(s + 1) * 8000],
                                                      win_len=8000)
                           for s in range(3)])


def extract_features_pcm(pcm_int16):
    """板端入口: 16bit PCM(int16) -> 295维特征"""
    return extract_features_from_wave(pcm_int16.astype(np.float64) / 32768.0)


if __name__ == "__main__":
    # 冒烟测试: 随机噪声 + 特征维度检查
    np.random.seed(0)
    y = np.random.randn(30000) * 0.01
    f = extract_features_from_wave(y)
    print("特征维度:", f.shape, " 无nan:", not np.isnan(f).any())
