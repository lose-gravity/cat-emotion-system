/* i2s_tx_v2.c - MAX98357A播放模块(s_axilite数据通道版, v2修复版)
 * 数据通道: pcm_packed[12000]走s_axilite寄存器, 每寄存器2个int16样本
 *           = 24000点 = 1.5s@16kHz音频
 * 输出: sd_out/bclk/ws/done为纯输出引脚(ap_none), 内部变量保持状态
 * 时钟: 相位累加器生成SCK=1.024MHz, WS=16kHz, 音频在左声道
 *
 * v2修复(相对v1):
 *   1. bclk不再读回引脚(*bclk=~(*bclk)会被HLS拆成bclk_i/bclk_o两个口),
 *      改用内部变量bclk_val ^= 1翻转
 *   2. ch = ~ch在32位无符号下会翻成0xFFFFFFFF导致采样不再装载,
 *      改用ch ^= 1
 */
void i2s_tx_v2(const int pcm_packed[12000], volatile unsigned int *sd_out,
               volatile unsigned int *bclk, volatile unsigned int *ws,
               volatile unsigned int *done) {
#pragma HLS interface s_axilite port=pcm_packed bundle=ctrl
#pragma HLS interface ap_none port=sd_out
#pragma HLS interface ap_none port=bclk
#pragma HLS interface ap_none port=ws
#pragma HLS interface ap_none port=done
#pragma HLS interface s_axilite port=return bundle=ctrl

    static short buf[24000];
    unsigned int phase = 0;
    unsigned int n = 0;
    unsigned int bit_cnt = 0;
    unsigned int ch = 0;
    unsigned int shift = 0;
    unsigned int sd_val = 0;
    unsigned int bclk_val = 0;
    int i;

    /* 解包 */
    for (i = 0; i < 12000; i++) {
#pragma HLS pipeline
        buf[2 * i]     = (short)(pcm_packed[i] & 0xFFFF);
        buf[2 * i + 1] = (short)((pcm_packed[i] >> 16) & 0xFFFF);
    }

    *done = 0;
    *ws = 1;
    *bclk = 0;
    *sd_out = 0;
    bclk_val = 0;

    play_loop:
    while (n < 24000) {
#pragma HLS pipeline II=1
        phase += 2048000;
        if (phase >= 100000000) {
            phase -= 100000000;
            bclk_val ^= 1;              /* 0/1之间翻转, 不用按位取反 */
            *bclk = bclk_val;

            if (bclk_val == 1) {
                bit_cnt++;
                if (bit_cnt == 32) {
                    bit_cnt = 0;
                    shift = 0;
                    ch ^= 1;
                    *ws = (ch == 0) ? 0 : 1;
                    if (ch == 0) {
                        shift = (unsigned int)(unsigned short)buf[n];
                        n++;
                    } else {
                        shift = 0;
                    }
                }
                if (ch == 0 && bit_cnt >= 2 && bit_cnt <= 17) {
                    sd_val = (shift >> (17 - bit_cnt)) & 1;
                } else {
                    sd_val = 0;
                }
                *sd_out = sd_val;
            }
        }
    }
    *done = 1;
}
