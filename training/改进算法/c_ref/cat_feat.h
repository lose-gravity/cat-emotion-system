/* cat_feat.h - 猫叫特征提取 (纯C99参考实现)
 *
 * 与 Python 版 features_numpy.py 完全一致的算法:
 *   输入: 24000点 int16 PCM (16kHz, 1.5s固定窗口)
 *   输出: 295维特征 (FeatBuf.feat)
 *   中间量 (mel_db/mfcc/d1/d2/centroid/rms/zcr) 供对拍调试
 *
 * 编译: gcc main.c cat_feat.c cat_mlp.c -lm -O2 -o cat_test.exe
 * 运行: cat_test.exe <文件名前缀>   例: cat_test.exe F_MAG01_EU_FN_FED01_106
 */
#ifndef CAT_FEAT_H
#define CAT_FEAT_H

#include <stdint.h>

#define SR         16000
#define N_FFT      2048
#define N_HALF     1025
#define WIN        400      /* 25ms MFCC窗 */
#define HOP        160      /* 10ms MFCC帧移 */
#define N_MFCC     40
#define WIN_LEN    24000    /* 1.5s固定窗口 */
#define HOP_RMS    512      /* rms/zcr/质心的帧移(注意与MFCC不同!) */
#define NFRAMES    151      /* 1 + 24000/160 */
#define NFRAMES_R  47       /* 1 + 24000/512 */
#define DIM        295

typedef struct {
    double re, im;
} cplx;

/* 特征缓冲区: 最终特征 + 各阶段中间量 */
typedef struct {
    double mel_db[NFRAMES][128];      /* log-mel谱, 帧x128 */
    double mfcc[40][NFRAMES];         /* DCT后40维 */
    double d1[40][NFRAMES];           /* delta一阶 */
    double d2[40][NFRAMES];           /* delta二阶 */
    double centroid[NFRAMES_R];       /* 谱质心(47帧) */
    double rms[NFRAMES_R];            /* RMS */
    double zcr[NFRAMES_R];            /* 过零率 */
    double feat[DIM];                 /* 295维最终特征 */
} FeatBuf;

/* 主入口: 24000点int16 PCM -> 特征+中间量 */
void extract_features(const int16_t *pcm, FeatBuf *out);

/* HLS顶层: 无结构体参数, 只出295维特征 */
void extract_features_top(const int16_t pcm[WIN_LEN], double feat[DIM]);

/* 分段特征: 24000点切成3段0.5s, 各提295维 -> 885维(集成模型2用) */
void extract_segmental(const int16_t *pcm, double feat885[3 * DIM]);

#endif
