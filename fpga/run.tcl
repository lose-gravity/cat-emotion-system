# run.tcl - 猫情绪识别特征提取 HLS工程脚本 (Vitis HLS 2022.1, float版)
# 用法: 在 Vitis HLS GUI 打开或命令行执行:
#   D:\vivado\Vitis_HLS\2022.1\bin\vitis_hls -f run.tcl
# 阶段说明: csim=C仿真(验证数值) csynth=综合(看资源/时序)
# 想只看仿真就注释掉 csynth_design 一行

open_project cat_feat_proj -reset
set_top cat_feat_ip_v3
add_files src/cat_feat_float.c
add_files src/cat_feat_ip.c
add_files src/hls_top.h
add_files src/tables_float_hls.h
add_files -tb tb/tb_ip.c

open_solution sol1 -reset
set_part {xc7z020clg400-1}          ;# PYNQ-Z1 / PYNQ-Z2 的芯片
create_clock -period 10             ;# 100MHz

csim_design -clean                   ;# C仿真: 3段金标准对拍, 应全部PASS
csynth_design                        ;# 综合: 生成资源/时序报告
# cosim_design -argv {B_DAK01_MC_FN_SIM01_102}   ;# RTL协同仿真: 可选用, XSIM警告刷屏极慢(5h+), 验证改到M1板级做
exit
