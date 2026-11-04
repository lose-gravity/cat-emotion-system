# build_overlay_v5.tcl - 特征IP + 舵机PWM + 喇叭I2S (s_axilite全通路)
# 结构: PS7 -> GP0 -> axi_gp_intercon -> cat_feat_ip_v3 (M00)
#                                     -> pwm_servo    (M01)
#                                     -> i2s_tx_v2    (M02)
# HLS的32位ap_none输出口用xlslice取bit0接到单bit外部引脚
# 引脚(Pmod JA): JA1=Y18 舵机PWM | JA2=Y19 喇叭BCLK | JA3=Y16 喇叭WS
#                JA4=Y17 喇叭DIN | JA7=W18 播放完成指示
set_param general.maxThreads 4
set part    "xc7z020clg400-1"
set board   "www.digilentinc.com:pynq-z1:part0:1.0"
set ipdir   [file normalize "./ip_repo"]
set pwmdir  [file normalize "./ip_repo_pwm"]
set i2sdir  [file normalize "./ip_repo_i2s"]
set outdir  [file normalize "."]

create_project cat_overlay_proj ./cat_overlay_proj -part $part -force
set_property board_part $board [current_project]
set_property ip_repo_paths [list $ipdir $pwmdir $i2sdir] [current_project]
update_ip_catalog

create_bd_design "cat_overlay_bd"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7]

create_bd_cell -type ip -vlnv xilinx.com:hls:cat_feat_ip_v3:1.0 cat_feat_ip_0
create_bd_cell -type ip -vlnv xilinx.com:hls:pwm_servo:1.0 pwm_servo_0
create_bd_cell -type ip -vlnv xilinx.com:hls:i2s_tx_v2:1.0 i2s_tx_0

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins rst_ps7/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_ps7/ext_reset_in]

foreach ip {cat_feat_ip_0 pwm_servo_0 i2s_tx_0} {
    connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins $ip/ap_clk]
    connect_bd_net [get_bd_pins rst_ps7/peripheral_aresetn] [get_bd_pins $ip/ap_rst_n]
}

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_gp_intercon
set_property -dict [list CONFIG.NUM_MI {3} CONFIG.NUM_SI {1}] [get_bd_cells axi_gp_intercon]
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_gp_intercon/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_gp_intercon/M00_AXI] [get_bd_intf_pins cat_feat_ip_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins axi_gp_intercon/M01_AXI] [get_bd_intf_pins pwm_servo_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins axi_gp_intercon/M02_AXI] [get_bd_intf_pins i2s_tx_0/s_axi_ctrl]
foreach p {ACLK S00_ACLK M00_ACLK M01_ACLK M02_ACLK} {
    connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_gp_intercon/$p]
}
foreach p {ARESETN S00_ARESETN M00_ARESETN M01_ARESETN M02_ARESETN} {
    connect_bd_net [get_bd_pins rst_ps7/peripheral_aresetn] [get_bd_pins axi_gp_intercon/$p]
}
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]

# 舵机PWM的en使能脚拉高(常使能)
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 en_const
set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {32}] [get_bd_cells en_const]
connect_bd_net [get_bd_pins en_const/dout] [get_bd_pins pwm_servo_0/en]

# HLS的32位输出口取bit0 -> 单bit外部引脚
proc slice_bit0 {name ip pin port} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 $name
    set_property -dict [list CONFIG.DIN_WIDTH {32} CONFIG.DIN_FROM {0} \
        CONFIG.DIN_TO {0} CONFIG.DOUT_WIDTH {1}] [get_bd_cells $name]
    connect_bd_net [get_bd_pins $ip/$pin] [get_bd_pins $name/Din]
    create_bd_port -dir O $port
    connect_bd_net [get_bd_pins $name/Dout] [get_bd_ports $port]
}
slice_bit0 slice_pwm  pwm_servo_0 pwm_out pwm_out
slice_bit0 slice_bclk i2s_tx_0    bclk    bclk_out
slice_bit0 slice_ws   i2s_tx_0    ws      ws_out
slice_bit0 slice_sd   i2s_tx_0    sd_out  sd_out
slice_bit0 slice_done i2s_tx_0    done    done_out

# 地址段: Vivado对AXI-Lite从机的段分配硬性封顶64K, 且Tcl不可改
# (set_property range/offset均"参数不存在", delete_bd_objs对seg无效)。
# 但特征IP的PCM寄存器在0x10000偏移, 需要128K段。
# 解法(probe_seg5验证通过): 保存BD后文本级修改.bd里的段定义, 重开BD即生效。
assign_bd_address
save_bd_design
close_bd_design [get_bd_designs cat_overlay_bd]

set bdpath ./cat_overlay_proj/cat_overlay_proj.srcs/sources_1/bd/cat_overlay_bd/cat_overlay_bd.bd
set fh [open $bdpath r]
set txt [read $fh]
close $fh
set p_cat [string first "\"SEG_cat_feat_ip_0_Reg\": \{" $txt]
set p_i2s [string first "\"SEG_i2s_tx_0_Reg\": \{" $txt]
set p_pwm [string first "\"SEG_pwm_servo_0_Reg\": \{" $txt]
if {$p_cat < 0 || $p_i2s < 0 || $p_pwm < 0} { error "找不到地址段记录" }
# cat_feat段: 64K -> 128K
set blk [string range $txt $p_cat [expr {$p_i2s - 1}]]
set blk [string map [list {"range": "64K"} {"range": "128K"}] $blk]
set txt [string replace $txt $p_cat [expr {$p_i2s - 1}] $blk]
# i2s段: 后移到0x40020000
set p_i2s [string first "\"SEG_i2s_tx_0_Reg\": \{" $txt]
set p_pwm [string first "\"SEG_pwm_servo_0_Reg\": \{" $txt]
set blk [string range $txt $p_i2s [expr {$p_pwm - 1}]]
set blk [string map [list {"offset": "0x40010000"} {"offset": "0x40020000"}] $blk]
set txt [string replace $txt $p_i2s [expr {$p_pwm - 1}] $blk]
# pwm段: 后移到0x40030000
set p_pwm [string first "\"SEG_pwm_servo_0_Reg\": \{" $txt]
set blk [string range $txt $p_pwm [expr {$p_pwm + 400}]]
set blk [string map [list {"offset": "0x40020000"} {"offset": "0x40030000"}] $blk]
set txt [string replace $txt $p_pwm [expr {$p_pwm + 400}] $blk]
set fh [open $bdpath w]
puts -nonewline $fh $txt
close $fh
puts "地址段已文本修正: cat_feat 128K, i2s 0x40020000, pwm 0x40030000"

open_bd_design [get_files cat_overlay_bd.bd]
set off_i2s ""
set off_pwm ""
foreach s [get_bd_addr_segs] {
    set nm [get_property name $s]
    set off [get_property offset $s]
    if {$off ne ""} { puts "seg: $nm off=$off" }
    if {$nm eq "SEG_i2s_tx_0_Reg"} { set off_i2s $off }
    if {$nm eq "SEG_pwm_servo_0_Reg"} { set off_pwm $off }
}
if {$off_i2s ne "0x40020000"} { error "i2s段修正失败: $off_i2s" }
if {$off_pwm ne "0x40030000"} { error "pwm段修正失败: $off_pwm" }
puts "地址段验证通过: i2s=$off_i2s pwm=$off_pwm"
save_bd_design
catch {validate_bd_design}

# 引脚约束: Pmod JA
set xdc [open "servo_pins.xdc" w]
puts $xdc "set_property PACKAGE_PIN Y18 \[get_ports pwm_out\]"   ;# JA1 舵机信号
puts $xdc "set_property PACKAGE_PIN Y19 \[get_ports bclk_out\]"  ;# JA2 喇叭BCLK
puts $xdc "set_property PACKAGE_PIN Y16 \[get_ports ws_out\]"    ;# JA3 喇叭WS/LRC
puts $xdc "set_property PACKAGE_PIN Y17 \[get_ports sd_out\]"    ;# JA4 喇叭DIN
puts $xdc "set_property PACKAGE_PIN W18 \[get_ports done_out\]"  ;# JA7 播放完成指示
foreach p {pwm_out bclk_out ws_out sd_out done_out} {
    puts $xdc "set_property IOSTANDARD LVCMOS33 \[get_ports $p\]"
}
close $xdc
add_files -fileset constrs_1 servo_pins.xdc

make_wrapper -files [get_files cat_overlay_bd.bd] -top
add_files -norecurse ./cat_overlay_proj/cat_overlay_proj.srcs/sources_1/bd/cat_overlay_bd/hdl/cat_overlay_bd_wrapper.v
update_compile_order -fileset sources_1

# 带重试的构建: IP生成的"slave interpreter"错误是Windows下的瞬时故障, 自动重试。
# 注意: 重试前只reset_run, 绝不能删.gen/ip目录(launch_runs不会重新填充,
# 会导致OOC综合run报"File does not exist" — v5j已踩过这个坑)
set bitfile ./cat_overlay_proj/cat_overlay_proj.runs/impl_1/cat_overlay_bd_wrapper.bit
set ok 0
for {set attempt 1} {$attempt <= 4} {incr attempt} {
    puts "===== 构建尝试 $attempt/4 ====="
    if {$attempt > 1} {
        catch {reset_run {*}[get_runs]}
    }
    if {[catch {launch_runs impl_1 -to_step write_bitstream -jobs 4} err]} {
        puts "launch_runs 失败(尝试$attempt): $err"
        continue
    }
    set status [wait_on_run impl_1]
    if {[file exists $bitfile]} {
        set ok 1
        break
    }
    puts "尝试 $attempt 未产出bitstream (status=$status), 重试..."
}
if {!$ok} { error "4次尝试后仍未产出 bitstream, 构建失败" }
file copy -force $bitfile $outdir/cat_overlay.bit
file copy -force ./cat_overlay_proj/cat_overlay_proj.gen/sources_1/bd/cat_overlay_bd/hw_handoff/cat_overlay_bd.hwh $outdir/cat_overlay.hwh
puts "OVERLAY V5 DONE"
