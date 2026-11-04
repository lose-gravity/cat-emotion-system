/* hls_top.h - HLS顶层接口定义(float版) */
#ifndef HLS_TOP_H
#define HLS_TOP_H

#define WIN_LEN 24000   /* 输入PCM点数(1.5s @16kHz) */
#define DIM     295     /* 单窗口特征维度 */
#define DIM_OUT 1180    /* 输出特征维度: 295(整窗) + 885(3段0.5s分段) */

/* 顶层: 24000点int16 PCM -> 295维float特征 */
void extract_features_top(const short pcm[WIN_LEN], float feat[DIM]);

/* 顶层v2: 同时输出295(整窗)+885(分段) -> feat[1180]
 * feat[0:295]   = 1.5s整窗特征(模型1用)
 * feat[295:1180]= 3段0.5s切片特征拼接(模型2用, 集成推理) */
void extract_features_top_v2(const short pcm[WIN_LEN], float feat[DIM_OUT]);

/* 上板顶层: AXI封装版v2(m_axi读写DDR + s_axilite控制) */
void cat_feat_ip(volatile int *pcm, volatile float *feat);

/* 上板顶层v3: 全部走s_axilite(板子实测GP0通路可靠)
 * pcm_packed[12000]: 每寄存器打包2个int16样本(lo|hi<<16)
 * feat_out[1180]: 特征输出(295整窗+885分段) */
void cat_feat_ip_v3(const int pcm_packed[12000], float feat_out[DIM_OUT]);

#endif
