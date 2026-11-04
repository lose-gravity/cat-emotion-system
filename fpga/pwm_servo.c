/* pwm_servo.c - SG90舵机PWM生成(练手模块, 纯C HLS)
 *
 * SG90 控制: 50Hz 周期(20ms), 脉宽 1.0~2.0ms 对应 0°~180°
 * 参数单位: 0.1us, 即 width=15000 表示 1.5ms(90°), period=200000 表示 20ms
 * 时钟: 100MHz(10ns), 计数一拍 = 0.1us, 不用换算
 *
 * PS 端用法(PYNQ Python):
 *   pwm = overlay.pwm_servo_0
 *   pwm.write(0x18, 200000)   # 周期 20ms(写一次)
 *   pwm.write(0x10, 15000)    # 转 90°
 *   pwm.write(0x20, 1)        # 使能
 *
 * HLS 工程: 单独建工程, set_top pwm_servo, 时钟 10ns, part 选 xc7z020clg400-1
 * 综合后在 Vivado 里把 pwm_out 引脚约束到你要接舵机的 Pmod 脚
 */
void pwm_servo(unsigned int width_x01us, unsigned int period_x01us,
               volatile unsigned int *en, volatile unsigned int *pwm_out) {
#pragma HLS interface s_axilite port=width_x01us    bundle=ctrl
#pragma HLS interface s_axilite port=period_x01us   bundle=ctrl
#pragma HLS interface ap_none port=en
#pragma HLS interface ap_none port=pwm_out
#pragma HLS interface s_axilite port=return         bundle=ctrl

    unsigned int cnt = 0;
    unsigned int high;

    if (width_x01us > period_x01us)
        width_x01us = period_x01us;        /* 防呆: 脉宽不能超过周期 */
    high = width_x01us;

    pwm_loop:
    while (1) {
#pragma HLS pipeline II=1
        if (*en) {
            *pwm_out = (cnt < high) ? 1 : 0;
            cnt++;
            if (cnt >= period_x01us)
                cnt = 0;
        } else {
            *pwm_out = 0;
            cnt = 0;
        }
    }
}
