# export.tcl - 把综合好的特征IP导出为Vivado IP包
# 用法: D:\vivado\Vitis_HLS\2022.1\bin\vitis_hls -f export.tcl
# 产物: ./cat_feat_ip_export/ (Vivado里 IP Catalog -> Add Repository 指向它)
open_project cat_feat_proj
open_solution sol1
export_design -format ip_catalog -output ./cat_feat_ip_export
exit
