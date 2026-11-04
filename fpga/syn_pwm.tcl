open_project pwm_proj -reset
set_top pwm_servo
add_files src/pwm_servo.c
open_solution sol1 -reset
set_part {xc7z020clg400-1}
create_clock -period 10
csynth_design
exit
