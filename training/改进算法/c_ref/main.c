/* main.c - C参考实现对拍测试
 *
 * 用法: cat_test.exe <文件名前缀>
 *   例: cat_test.exe F_MAG01_EU_FN_FED01_106
 * 自动对比(相对本目录):
 *   ../golden/{base}_pcm.csv            输入(24000个int16)
 *   ../golden/interm/{base}/*.csv       各阶段中间量参考
 *   ../golden/{base}_features295.csv    295维特征参考
 *   ../golden/{base}_features885.csv    885维分段特征参考
 *   ../golden/{base}_probs.csv          集成概率参考
 *   ../weights_float_final/model1|2     模型权重
 * 通过标准: 特征误差<1e-3, 概率差<0.01
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "cat_feat.h"
#include "cat_mlp.h"

/* ---- CSV读取 ---- */
static double *read_csv_flat(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    double *data = NULL;
    int cnt = 0, cap = 0;
    if (!f) { fprintf(stderr, "无法打开: %s\n", path); return NULL; }
    for (;;) {
        int ch = fgetc(f);
        if (ch == EOF) break;
        if (ch == ',' || ch == '\n' || ch == '\r') {
            /* 分隔符: 结束一个数(空值跳过, 由调用方保证格式规整) */
            continue;
        }
        ungetc(ch, f);
        if (cnt >= cap) {
            cap = cap ? cap * 2 : 1024;
            data = (double *)realloc(data, (size_t)cap * sizeof(double));
        }
        fscanf(f, "%lf", &data[cnt]);
        cnt++;
    }
    fclose(f);
    *n = cnt;
    return data;
}

static int16_t *read_pcm(const char *path) {
    int n = 0;
    double *d = read_csv_flat(path, &n);
    int16_t *out;
    int i;
    if (!d || n != WIN_LEN) {
        fprintf(stderr, "PCM长度异常: %d (应为%d) %s\n", n, WIN_LEN, path);
        free(d);
        return NULL;
    }
    out = (int16_t *)malloc(n * sizeof(int16_t));
    for (i = 0; i < n; i++) out[i] = (int16_t)d[i];
    free(d);
    return out;
}

static double max_abs_diff(const double *a, const double *b, int n) {
    double m = 0;
    int i;
    for (i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *base;
    char path[1024];
    int16_t *pcm;
    static FeatBuf buf;
    double feat885[3 * DIM];
    MLP m1, m2;
    double logits[3], prob[3];
    int fail = 0, n;

    if (argc < 2) {
        printf("用法: cat_test.exe <文件名前缀>\n");
        return 1;
    }
    base = argv[1];

    /* 输入 */
    snprintf(path, sizeof(path), "../golden/%s_pcm.csv", base);
    pcm = read_pcm(path);
    if (!pcm) return 1;

    /* 特征 */
    extract_features(pcm, &buf);
    extract_segmental(pcm, feat885);

    /* 各阶段中间量对拍 */
    printf("==== %s ====\n", base);
#define STAGE(var, dim, fname, tol)                                           \
    do {                                                                      \
        double *ref; int nn;                                                  \
        snprintf(path, sizeof(path), "../golden/interm/%s/%s", base, fname);  \
        ref = read_csv_flat(path, &nn);                                       \
        if (ref && nn == (dim)) {                                             \
            double e = max_abs_diff((const double *)(var), ref, (dim));       \
            printf("  %-22s 最大误差 %.3e  %s\n", fname, e,                    \
                   e < (tol) ? "OK" : "FAIL");                                 \
            if (e >= (tol)) fail = 1;                                         \
        } else {                                                              \
            printf("  %-22s 参考缺失或维度不符(%d)\n", fname, nn);            \
            fail = 1;                                                         \
        }                                                                     \
        free(ref);                                                            \
    } while (0)

    STAGE(buf.mel_db, NFRAMES * 128, "mel_db_151x128.csv", 1e-3);
    STAGE(buf.mfcc, 40 * NFRAMES, "mfcc_40x151.csv", 1e-3);
    STAGE(buf.d1, 40 * NFRAMES, "d1_40x151.csv", 1e-3);
    STAGE(buf.d2, 40 * NFRAMES, "d2_40x151.csv", 1e-3);
    STAGE(buf.centroid, NFRAMES_R, "centroid_47.csv", 1e-3);
    STAGE(buf.rms, NFRAMES_R, "rms_47.csv", 1e-3);
    STAGE(buf.zcr, NFRAMES_R, "zcr_47.csv", 1e-3);
#undef STAGE

    /* 295/885维最终特征 */
    {
        double *ref;
        snprintf(path, sizeof(path), "../golden/%s_features295.csv", base);
        ref = read_csv_flat(path, &n);
        if (ref && n == DIM) {
            double e = max_abs_diff(buf.feat, ref, DIM);
            printf("  %-22s 最大误差 %.3e  %s\n", "features295", e,
                   e < 1e-3 ? "OK" : "FAIL");
            if (e >= 1e-3) fail = 1;
        }
        free(ref);
        snprintf(path, sizeof(path), "../golden/%s_features885.csv", base);
        ref = read_csv_flat(path, &n);
        if (ref && n == 3 * DIM) {
            double e = max_abs_diff(feat885, ref, 3 * DIM);
            printf("  %-22s 最大误差 %.3e  %s\n", "features885", e,
                   e < 1e-3 ? "OK" : "FAIL");
            if (e >= 1e-3) fail = 1;
        }
        free(ref);
    }

    /* 集成推理 */
    if (mlp_load(&m1, "../weights_float_final/model1")) return 1;
    if (mlp_load(&m2, "../weights_float_final/model2")) return 1;
    {
        double l1[3], p1[3], l2[3], p2[3];
        mlp_forward(&m1, buf.feat, l1, p1);
        mlp_forward(&m2, feat885, l2, p2);
        for (n = 0; n < 3; n++) {
            prob[n] = (p1[n] + p2[n]) / 2.0;
            logits[n] = (l1[n] + l2[n]) / 2.0;
        }
        {
            double *ref;
            snprintf(path, sizeof(path), "../golden/%s_probs.csv", base);
            ref = read_csv_flat(path, &n);
            if (ref && n == 3) {
                double e = max_abs_diff(prob, ref, 3);
                printf("  %-22s 最大误差 %.3e  %s\n", "probs(集成)", e,
                       e < 0.01 ? "OK" : "FAIL");
                if (e >= 0.01) fail = 1;
                printf("  概率: 亲昵%.3f 饥饿%.3f 焦虑%.3f\n",
                       prob[0], prob[1], prob[2]);
                if (prob[1] >= 0.5)
                    printf("  >>> 触发: 舵机出粮\n");
                else if (prob[2] >= 0.6)
                    printf("  >>> 触发: 逗猫棒+安抚音频\n");
                else
                    printf("  >>> 不动作\n");
            }
            free(ref);
        }
    }
    mlp_free(&m1);
    mlp_free(&m2);
    free(pcm);

    printf("%s\n", fail ? "结论: FAIL ✗ (有阶段超差, 见上)" : "结论: 全部通过 ✓");
    return fail;
}
