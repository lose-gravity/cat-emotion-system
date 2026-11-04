/* cat_feat_ip.c - 特征提取IP的AXI封装(上板用顶层)
 *
 * 把裸数组顶层包成 PS 可调用的 IP:
 *   - pcm : m_axi 接口, PS 在 DDR 里放 24000 个 int32(每个元素一个样本, 值域-32768~32767)
 *   - feat: m_axi 接口, IP 算完把 295 个 float 写回 DDR
 *   - 控制: s_axilite(启动/完成状态), PS 用 PYNQ 的 mmio 读写
 *
 * PS 侧用法(Python):
 *   pcm_buf = allocate(shape=(24000,), dtype=np.int32)   # 填好音频
 *   feat_buf = allocate(shape=(295,), dtype=np.float32)
 *   ip.write(0x10, pcm_buf.physical_address)
 *   ip.write(0x18, feat_buf.physical_address)
 *   ip.write(0x00, 1)                     # ap_start
 *   while ip.read(0x00) & 0x2 == 0: pass  # 等 ap_done
 *   feat = np.array(feat_buf)
 *
 * 注意: 本文件与 cat_feat_float.c 一起编译; run.tcl 里 set_top 改成本函数
 */
#include "hls_top.h"

void cat_feat_ip(volatile int *pcm, volatile float *feat) {
#pragma HLS interface m_axi port=pcm  offset=slave bundle=gmem depth=24000
#pragma HLS interface m_axi port=feat offset=slave bundle=gmem depth=1180
#pragma HLS interface s_axilite port=return bundle=ctrl
    static short buf[WIN_LEN];
    static float f[DIM_OUT];
    int i;

    /* DDR -> 本地缓冲 (int32 每元素一个样本) */
    for (i = 0; i < WIN_LEN; i++) {
#pragma HLS pipeline
        buf[i] = (short)pcm[i];
    }

    /* 特征提取v2: 295(整窗) + 885(3段切片) = 1180维 */
    extract_features_top_v2(buf, f);

    /* 写回 DDR */
    for (i = 0; i < DIM_OUT; i++) {
#pragma HLS pipeline
        feat[i] = f[i];
    }
}

/* v3: 全s_axilite数据通道(板上实测GP0路径可靠, 不依赖HP0/m_axi)
 * 注意: 必须用数组参数(不是指针), 才能映射成s_axilite寄存器块
 * 寄存器布局(HLS自动): feat_out 0x2000, pcm_packed 0x10000 —
 * HLS要求s_axilite数组按"大小取整到2的幂"对齐, 无法压进64K,
 * 超段问题在BD侧解决(build_overlay_v5.tcl里文本修正段范围为128K) */
void cat_feat_ip_v3(const int pcm_packed[12000], float feat_out[DIM_OUT]) {
#pragma HLS interface s_axilite port=pcm_packed bundle=ctrl
#pragma HLS interface s_axilite port=feat_out bundle=ctrl
#pragma HLS interface s_axilite port=return bundle=ctrl
    static short buf[WIN_LEN];
    static float f[DIM_OUT];
    int i;

    /* 解包: 每寄存器2个int16样本 */
    for (i = 0; i < 12000; i++) {
#pragma HLS pipeline
        buf[2 * i]     = (short)(pcm_packed[i] & 0xFFFF);
        buf[2 * i + 1] = (short)((pcm_packed[i] >> 16) & 0xFFFF);
    }

    extract_features_top_v2(buf, f);

    for (i = 0; i < DIM_OUT; i++) {
#pragma HLS pipeline
        feat_out[i] = f[i];
    }
}
