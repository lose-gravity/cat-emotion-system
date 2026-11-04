open_project i2s_tx_proj -reset
set_top i2s_tx_v2
add_files src/i2s_tx_v2.c
open_solution sol1 -reset
set_part {xc7z020clg400-1}
create_clock -period 10
csynth_design
exit
