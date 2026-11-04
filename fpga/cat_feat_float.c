/* cat_feat_float.c - HLS用float版特征提取 v2(资源优化版)
 * 算法与double版cat_feat.c完全一致, 数学等价的资源优化:
 *   1. 不用ypad/ye大数组: 补零/边缘填充改为"带钳位的下标读原始PCM"
 *      (zero-pad: 越界取0; edge-pad: 越界取边界值)
 *   2. 不存sg矩阵/d1/d2数组: delta均值 = mfcc与sg矩阵"列和"的点积
 *      (矩阵乘再求均值 == 列和点积, 数学恒等)
 *   3. 全部float
 * 数值容差: 对golden的295维特征误差 <5e-3 (对模型概率影响实测为0)
 */
#include "hls_top.h"
#include "tables_float_hls.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ---------- 基2 FFT (迭代, 位反转) ---------- */
typedef struct { float re, im; } cplx_f;

static void bit_reverse(cplx_f *a, int n) {
    int i, j;
    for (i = 1, j = 0; i < n; i++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline
#endif
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { cplx_f t = a[i]; a[i] = a[j]; a[j] = t; }
    }
}

static void fft_radix2(cplx_f *a, int n) {
    int len, i, k;
    bit_reverse(a, n);
    for (len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / (float)len;
        cplx_f wlen;
        wlen.re = cosf(ang);
        wlen.im = sinf(ang);
        for (i = 0; i < n; i += len) {
            cplx_f w = { 1.0f, 0.0f };
            for (k = 0; k < len / 2; k++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline
#endif
                cplx_f u = a[i + k];
                cplx_f t = a[i + k + len / 2];
                cplx_f v;
                v.re = t.re * w.re - t.im * w.im;
                v.im = t.re * w.im + t.im * w.re;
                a[i + k].re = u.re + v.re;
                a[i + k].im = u.im + v.im;
                a[i + k + len / 2].re = u.re - v.re;
                a[i + k + len / 2].im = u.im - v.im;
                {
                    float nw_re = w.re * wlen.re - w.im * wlen.im;
                    float nw_im = w.re * wlen.im + w.im * wlen.re;
                    w.re = nw_re;
                    w.im = nw_im;
                }
            }
        }
    }
}

/* ---------- 百分位数(numpy linear) + 统计 ---------- */
static float percentile_linear(float *x, int n, float p) {
    float idx = (n - 1) * p / 100.0f;
    int lo = (int)floorf(idx), hi = (int)ceilf(idx);
    float w = idx - (float)lo;
    return x[lo] * (1.0f - w) + x[hi] * w;
}

static void stats5(float *x, int n, float *out) {
    int i, j;
    float mean = 0, var = 0;
    for (i = 1; i < n; i++) {          /* 插入排序(可综合) */
        float key = x[i];
        for (j = i - 1; j >= 0 && x[j] > key; j--) {
#ifdef __SYNTHESIS__
#pragma HLS loop_tripcount min=0 max=151 avg=75
#endif
            x[j + 1] = x[j];
        }
        x[j + 1] = key;
    }
    out[0] = percentile_linear(x, n, 10.0f);
    out[1] = percentile_linear(x, n, 50.0f);
    out[2] = percentile_linear(x, n, 90.0f);
    for (i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    out[3] = mean;
    out[4] = sqrtf(var / (float)n);
}

/* ---------- 核心特征提取(任意窗口长度n, 151或51帧) ---------- */
static void core_extract(const short *pcm, int n, const float *cs1, const float *cs2,
                         float feat[DIM]) {
    static cplx_f buf[2048];
    static float mel_db[151][128];
    static float mfcc[40][151];
    static float win_full[2048];
    static float cent[47], rmsv[47], zcrv[47];
    static float row[151];
    static int win_init = 0;

    int nf = 1 + n / 160;    /* 24000->151帧, 8000->51帧 */
    int nr = 1 + n / 512;    /* 24000->47帧, 8000->16帧 */
    int i, f, k, r, j;
    float gmax;

    if (!win_init) {
        memset(win_full, 0, sizeof(win_full));
        for (i = 0; i < 400; i++)
            win_full[(2048 - 400) / 2 + i] = (float)hann400[0][i];
        win_init = 1;
    }

    /* ---- 1. MFCC: FFT -> 功率谱 -> mel -> log -> top_db截断 ---- */
    /* 帧f中心在样本 f*160, 窗口覆盖 [f*160-1024, f*160+1024): 越界补零 */
    for (f = 0; f < nf; f++) {
        int base = f * 160 - 1024;
        for (i = 0; i < 2048; i++) {
            int g = base + i;
            float v = (g >= 0 && g < n) ? (float)pcm[g] / 32768.0f : 0.0f;
            buf[i].re = v * win_full[i];
            buf[i].im = 0.0f;
        }
        fft_radix2(buf, 2048);
        for (r = 0; r < 128; r++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline
#endif
            float acc = 0.0f;
            for (j = 0; j < mel_nz[r]; j++) {
#ifdef __SYNTHESIS__
#pragma HLS loop_tripcount min=5 max=48 avg=24
#endif
                k = mel_col[r][j];
                acc += (float)mel_val[r][j] *
                       (buf[k].re * buf[k].re + buf[k].im * buf[k].im);
            }
            mel_db[f][r] = 10.0f * log10f(fmaxf(acc, 1e-10f));
        }
    }
    gmax = -1e30f;
    for (f = 0; f < nf; f++)
        for (r = 0; r < 128; r++)
            if (mel_db[f][r] > gmax) gmax = mel_db[f][r];
    for (f = 0; f < nf; f++)
        for (r = 0; r < 128; r++)
            if (mel_db[f][r] < gmax - 80.0f) mel_db[f][r] = gmax - 80.0f;

    /* ---- 2. DCT-II(ortho) ---- */
    for (k = 0; k < 40; k++)
        for (f = 0; f < nf; f++) {
            float acc = 0.0f;
            for (r = 0; r < 128; r++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline II=1
#endif
                acc += (float)dct40[k][r] * mel_db[f][r];
            }
            mfcc[k][f] = acc;
        }

    /* ---- 3. 谱质心(幅度谱, Hann2048, 帧移512, 越界补零) ---- */
    for (f = 0; f < nr; f++) {
        float num = 0.0f, den = 0.0f;
        int base = f * 512 - 1024;
        for (i = 0; i < 2048; i++) {
            int g = base + i;
            float v = (g >= 0 && g < n) ? (float)pcm[g] / 32768.0f : 0.0f;
            buf[i].re = v * (float)hann2048[0][i];
            buf[i].im = 0.0f;
        }
        fft_radix2(buf, 2048);
        for (k = 0; k < 1025; k++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline
#endif
            float mag = sqrtf(buf[k].re * buf[k].re + buf[k].im * buf[k].im);
            float freq = 16000.0f * (float)k / 2048.0f;
            num += freq * mag;
            den += mag;
        }
        cent[f] = (den > 0.0f) ? num / den : 0.0f;
    }

    /* ---- 4. RMS(越界补零) ---- */
    for (f = 0; f < nr; f++) {
        float acc = 0.0f;
        int base = f * 512 - 1024;
        for (i = 0; i < 2048; i++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline II=1
#endif
            int g = base + i;
            float v = (g >= 0 && g < n) ? (float)pcm[g] / 32768.0f : 0.0f;
            acc += v * v;
        }
        rmsv[f] = sqrtf(acc / 2048.0f);
    }

    /* ---- 5. 过零率(边缘填充: 越界取边界值!) ---- */
    for (f = 0; f < nr; f++) {
        int crossings = 0, prev_neg;
        int base = f * 512 - 1024;
        float v0;
        {
            int g = base < 0 ? 0 : (base >= n ? n - 1 : base);
            v0 = (float)pcm[g] / 32768.0f;
        }
        if (fabsf(v0) <= 1e-10f) v0 = 0.0f;
        prev_neg = (v0 < 0.0f) ? 1 : 0;
        for (i = 1; i < 2048; i++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline
#endif
            int g = base + i;
            int gc = g < 0 ? 0 : (g >= n ? n - 1 : g);
            float v = (float)pcm[gc] / 32768.0f;
            int neg;
            if (fabsf(v) <= 1e-10f) v = 0.0f;
            neg = (v < 0.0f) ? 1 : 0;
            if (neg != prev_neg) crossings++;
            prev_neg = neg;
        }
        zcrv[f] = (float)crossings / 2048.0f;
    }

    /* ---- 6. delta均值: 用sg矩阵列和做点积(数学恒等, 免存sg矩阵/d1/d2) ---- */
    /* ---- 7. 统计量组装295维 ---- */
    {
        float s[5];
        int o = 0;
        for (k = 0; k < 40; k++) {
            memcpy(row, mfcc[k], nf * sizeof(float));
            stats5(row, nf, s);
            memcpy(feat + o, s, 5 * sizeof(float));
            o += 5;
        }
        for (k = 0; k < 40; k++) {
            float m = 0.0f;
            for (f = 0; f < nf; f++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline II=1
#endif
                m += cs1[f] * mfcc[k][f];
            }
            feat[o++] = m / (float)nf;
        }
        for (k = 0; k < 40; k++) {
            float m = 0.0f;
            for (f = 0; f < nf; f++) {
#ifdef __SYNTHESIS__
#pragma HLS pipeline II=1
#endif
                m += cs2[f] * mfcc[k][f];
            }
            feat[o++] = m / (float)nf;
        }
        memcpy(row, cent, nr * sizeof(float));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(float)); o += 5;
        memcpy(row, rmsv, nr * sizeof(float));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(float)); o += 5;
        memcpy(row, zcrv, nr * sizeof(float));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(float)); o += 5;
    }
}

/* ---------- 公开顶层 ---------- */
void extract_features_top(const short pcm[WIN_LEN], float feat[DIM]) {
    core_extract(pcm, WIN_LEN, colsum_sg1, colsum_sg2, feat);
}

void extract_features_top_v2(const short pcm[WIN_LEN], float feat[DIM_OUT]) {
    int s;
    /* 整窗295维(模型1) */
    core_extract(pcm, WIN_LEN, colsum_sg1, colsum_sg2, feat);
    /* 3段0.5s切片各295维(模型2), 用51帧版列和 */
    for (s = 0; s < 3; s++)
        core_extract(pcm + s * 8000, 8000,
                     colsum_sg1_51, colsum_sg2_51, feat + DIM + s * DIM);
}
