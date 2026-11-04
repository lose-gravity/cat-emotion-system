/* cat_feat.c - 猫叫特征提取 (纯C99, 与features_numpy.py对拍一致)
 *
 * 算法步骤(详见 ../特征规格.md):
 *   MFCC:  补零1024 -> 分帧2048/帧移160(151帧) -> 周期Hann400窗 -> 2048点FFT
 *          -> 功率谱 -> mel稀疏滤波(128) -> 10log10(max(x,1e-10)) -> top_db=80截断
 *          -> DCT-II(ortho)取40维
 *   delta: savgol滤波(width=9, deriv=order, mode=interp) = 查sg1/sg2矩阵
 *   质心:  幅度谱(帧移512, Hann2048窗, 47帧), 全零帧=0
 *   rms:   补零1024 -> 帧2048/帧移512 -> sqrt(mean(x^2))
 *   zcr:   边缘填充1024(复制边值!) -> 帧2048/帧移512 -> |x|<=1e-10归零后数符号翻转
 *   统计:  每维[p10,p50,p90,mean,std], 组装295维
 */
#include "cat_feat.h"
#include "tables.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- 基础: 基2 FFT (迭代, 位反转, 方向与numpy一致) ---------- */
static void bit_reverse(cplx *a, int n) {
    int i, j;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { cplx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
}

void fft_radix2(cplx *a, int n) {
    int len, i, k;
    bit_reverse(a, n);
    for (len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        cplx wlen = { cos(ang), sin(ang) };
        for (i = 0; i < n; i += len) {
            cplx w = { 1.0, 0.0 };
            for (k = 0; k < len / 2; k++) {
                cplx u = a[i + k];
                cplx t = a[i + k + len / 2];
                cplx v = { t.re * w.re - t.im * w.im,
                           t.re * w.im + t.im * w.re };
                a[i + k].re = u.re + v.re;
                a[i + k].im = u.im + v.im;
                a[i + k + len / 2].re = u.re - v.re;
                a[i + k + len / 2].im = u.im - v.im;
                double nw_re = w.re * wlen.re - w.im * wlen.im;
                double nw_im = w.re * wlen.im + w.im * wlen.re;
                w.re = nw_re;
                w.im = nw_im;
            }
        }
    }
}

/* ---------- 百分位数(numpy linear插值法) ---------- */
static double percentile_linear(double *x, int n, double p) {
    double idx = (n - 1) * p / 100.0;
    int lo = (int)floor(idx), hi = (int)ceil(idx);
    double w = idx - lo;
    return x[lo] * (1.0 - w) + x[hi] * w;
}

/* 对一维序列算[p10,p50,p90,mean,std], 结果写入out[0..4]
 * 注意: 排序用插入排序(不用qsort) —— Vitis HLS无法综合标准库qsort */
static void stats5(double *x, int n, double *out) {
    static double tmp[512];
    int i, j;
    double mean = 0, var = 0;
    memcpy(tmp, x, n * sizeof(double));
    for (i = 1; i < n; i++) {          /* 插入排序, 升序 */
        double key = tmp[i];
        for (j = i - 1; j >= 0 && tmp[j] > key; j--)
            tmp[j + 1] = tmp[j];
        tmp[j + 1] = key;
    }
    out[0] = percentile_linear(tmp, n, 10.0);
    out[1] = percentile_linear(tmp, n, 50.0);
    out[2] = percentile_linear(tmp, n, 90.0);
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    out[3] = mean;
    out[4] = sqrt(var / n);
}

/* ---------- 中间量文件级缓存(extract_features填, testbench读) ---------- */
static double g_mel_db[NFRAMES][128];
static double g_mfcc[40][NFRAMES];
static double g_d1[40][NFRAMES];
static double g_d2[40][NFRAMES];
static double g_cent[NFRAMES_R];
static double g_rms[NFRAMES_R];
static double g_zcr[NFRAMES_R];

/* ---------- 核心: 任意长度波形 -> 295维, 中间量同步保存 ---------- */
/* sg1/sg2: savgol矩阵指针(151x151或51x51, 行主序) */
static void core_extract(const double *y, int n, const double *win_full,
                         const double *sg1, const double *sg2,
                         int save_interm, double feat[DIM]) {
    static cplx buf[N_FFT];
    static double ypad[N_FFT + WIN_LEN + 16];
    static double ye[N_FFT + WIN_LEN + 16];
    static double mfcc[40][NFRAMES];
    static double mel_db[NFRAMES][128];
    int nf, nr, i, f, k, r, j, pad = N_FFT / 2;
    double gmax;
    double(*cent)[NFRAMES_R] = (double(*)[NFRAMES_R])g_cent;
    double(*rmsv)[NFRAMES_R] = (double(*)[NFRAMES_R])g_rms;
    double(*zcrv)[NFRAMES_R] = (double(*)[NFRAMES_R])g_zcr;

    nf = 1 + n / HOP;       /* 1.5s=151帧, 0.5s=51帧 */
    nr = 1 + n / HOP_RMS;   /* 1.5s=47帧, 0.5s=16帧 */

    memset(ypad, 0, sizeof(ypad));
    memcpy(ypad + pad, y, n * sizeof(double));

    /* ---- 1. MFCC: FFT -> 功率谱 -> mel稀疏滤波 -> log -> top_db截断 ---- */
    for (f = 0; f < nf; f++) {
        for (i = 0; i < N_FFT; i++) {
            buf[i].re = ypad[f * HOP + i] * win_full[i];
            buf[i].im = 0.0;
        }
        fft_radix2(buf, N_FFT);
        for (r = 0; r < 128; r++) {
            double acc = 0.0;
            for (j = 0; j < mel_nz[r]; j++) {
                k = mel_col[r][j];
                acc += mel_val[r][j] * (buf[k].re * buf[k].re + buf[k].im * buf[k].im);
            }
            mel_db[f][r] = 10.0 * log10(fmax(acc, 1e-10));
        }
    }
    gmax = -1e30;
    for (f = 0; f < nf; f++)
        for (r = 0; r < 128; r++)
            if (mel_db[f][r] > gmax) gmax = mel_db[f][r];
    for (f = 0; f < nf; f++)
        for (r = 0; r < 128; r++)
            if (mel_db[f][r] < gmax - 80.0) mel_db[f][r] = gmax - 80.0;

    /* ---- 2. DCT-II(ortho) ---- */
    for (k = 0; k < 40; k++)
        for (f = 0; f < nf; f++) {
            double acc = 0.0;
            for (r = 0; r < 128; r++) acc += dct40[k][r] * mel_db[f][r];
            mfcc[k][f] = acc;
        }

    /* ---- 3. 谱质心(幅度谱, Hann2048, 帧移512) ---- */
    for (f = 0; f < nr; f++) {
        double num = 0.0, den = 0.0;
        for (i = 0; i < N_FFT; i++) {
            buf[i].re = ypad[f * HOP_RMS + i] * hann2048[0][i];
            buf[i].im = 0.0;
        }
        fft_radix2(buf, N_FFT);
        for (k = 0; k < N_HALF; k++) {
            double mag = sqrt(buf[k].re * buf[k].re + buf[k].im * buf[k].im);
            double freq = (double)SR * k / N_FFT;
            num += freq * mag;
            den += mag;
        }
        cent[0][f] = (den > 0.0) ? num / den : 0.0;   /* 全零帧=0 */
    }

    /* ---- 4. RMS(补零constant) ---- */
    for (f = 0; f < nr; f++) {
        double acc = 0.0;
        for (i = 0; i < N_FFT; i++) {
            double v = ypad[f * HOP_RMS + i];
            acc += v * v;
        }
        rmsv[0][f] = sqrt(acc / N_FFT);
    }

    /* ---- 5. 过零率(边缘填充edge! 与RMS的constant不同) ---- */
    memcpy(ye, ypad, (pad + n + pad) * sizeof(double));
    for (i = 0; i < pad; i++) ye[i] = y[0];
    for (i = 0; i < pad; i++) ye[pad + n + i] = y[n - 1];
    for (f = 0; f < nr; f++) {
        int crossings = 0, prev_neg;
        double v0 = ye[f * HOP_RMS];
        if (fabs(v0) <= 1e-10) v0 = 0.0;
        prev_neg = (v0 < 0.0) ? 1 : 0;
        for (i = 1; i < N_FFT; i++) {
            double v = ye[f * HOP_RMS + i];
            int neg;
            if (fabs(v) <= 1e-10) v = 0.0;
            neg = (v < 0.0) ? 1 : 0;
            if (neg != prev_neg) crossings++;
            prev_neg = neg;
        }
        zcrv[0][f] = (double)crossings / N_FFT;
    }

    /* ---- 6. delta: d[k][f] = sum_t sg[f][t] * mfcc[k][t] ---- */
    for (k = 0; k < 40; k++)
        for (f = 0; f < nf; f++) {
            double a1 = 0.0, a2 = 0.0;
            for (j = 0; j < nf; j++) {
                a1 += sg1[f * nf + j] * mfcc[k][j];
                a2 += sg2[f * nf + j] * mfcc[k][j];
            }
            g_d1[k][f] = a1;
            g_d2[k][f] = a2;
        }

    /* 保存中间量(仅1.5s路径) */
    if (save_interm) {
        memcpy(g_mel_db, mel_db, sizeof(mel_db));
        memcpy(g_mfcc, mfcc, sizeof(mfcc));
    }

    /* ---- 7. 统计量组装295维 ---- */
    {
        double s[5];
        double row[NFRAMES];
        int o = 0;
        for (k = 0; k < 40; k++) {
            memcpy(row, mfcc[k], nf * sizeof(double));
            stats5(row, nf, s);
            memcpy(feat + o, s, 5 * sizeof(double));
            o += 5;
        }
        for (k = 0; k < 40; k++) {
            double m = 0.0;
            for (f = 0; f < nf; f++) m += g_d1[k][f];
            feat[o++] = m / nf;
        }
        for (k = 0; k < 40; k++) {
            double m = 0.0;
            for (f = 0; f < nf; f++) m += g_d2[k][f];
            feat[o++] = m / nf;
        }
        memcpy(row, cent[0], nr * sizeof(double));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(double)); o += 5;
        memcpy(row, rmsv[0], nr * sizeof(double));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(double)); o += 5;
        memcpy(row, zcrv[0], nr * sizeof(double));
        stats5(row, nr, s); memcpy(feat + o, s, 5 * sizeof(double)); o += 5;
    }
}

/* ---------- 公开入口 ---------- */
static double win_full[N_FFT];
static int win_init = 0;

static void ensure_win(void) {
    int i;
    if (win_init) return;
    memset(win_full, 0, sizeof(win_full));
    for (i = 0; i < WIN; i++)
        win_full[(N_FFT - WIN) / 2 + i] = hann400[0][i];
    win_init = 1;
}

void extract_features(const int16_t *pcm, FeatBuf *out) {
    static double y[WIN_LEN];
    int i;
    ensure_win();
    for (i = 0; i < WIN_LEN; i++) y[i] = pcm[i] / 32768.0;
    core_extract(y, WIN_LEN, win_full, &sg1[0][0], &sg2[0][0], 1, out->feat);
    memcpy(out->mel_db, g_mel_db, sizeof(g_mel_db));
    memcpy(out->mfcc, g_mfcc, sizeof(g_mfcc));
    memcpy(out->d1, g_d1, sizeof(g_d1));
    memcpy(out->d2, g_d2, sizeof(g_d2));
    memcpy(out->centroid, g_cent, sizeof(g_cent));
    memcpy(out->rms, g_rms, sizeof(g_rms));
    memcpy(out->zcr, g_zcr, sizeof(g_zcr));
}

void extract_features_top(const int16_t pcm[WIN_LEN], double feat[DIM]) {
    /* HLS专用顶层: 无结构体参数, 只出295维特征
     * (extract_features带FeatBuf中间量, HLS无法把结构体拆成端口) */
    static double y[WIN_LEN];
    int i;
    ensure_win();
    for (i = 0; i < WIN_LEN; i++) y[i] = pcm[i] / 32768.0;
    core_extract(y, WIN_LEN, win_full, &sg1[0][0], &sg2[0][0], 0, feat);
}

void extract_segmental(const int16_t *pcm, double feat885[3 * DIM]) {
    static double y[WIN_LEN];
    int s, i;
    ensure_win();
    for (i = 0; i < WIN_LEN; i++) y[i] = pcm[i] / 32768.0;
    for (s = 0; s < 3; s++)
        core_extract(y + s * 8000, 8000, win_full, &sg1_51[0][0], &sg2_51[0][0],
                     0, feat885 + s * DIM);
}
