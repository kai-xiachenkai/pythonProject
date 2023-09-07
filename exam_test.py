import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *



N = 4000
n = 1000


v_fr = 0.001
v_fb = 0.006
v_fg = 0.001
v_rr = 0.5
v_rb = 0.55*0.9
v_rg = 0.45*0.9
v_br = 0.55*0.9
v_bb = 0.45
v_bg = 0.60*0.9
v_gr = 0.45*0.9
v_gb = 0.60*0.9
v_gg = 0.6


pr = 0.25
pb = 0.25

vfr_list = []
rr_list = []
rb_list = []
rg_list = []
bb_list = []
bg_list = []
gg_list = []

start_time = datetime.now()

initial_list,red_list,blue_list,green_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                      v_bb, v_bg, v_gr, v_gb, v_gg)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


start_time = datetime.now()

total_rr, total_rb, total_rg, total_bb, total_bg, total_gg = search_loop(n, final_list)

end_time = datetime.now()

print('Duration: {}'.format(end_time - start_time))

print(total_rr,total_rb,total_rg,total_bb,total_bg,total_gg)


start_time = datetime.now()

total_rr, total_rb, total_rg, total_bb, total_bg, total_gg = search_loop1(n, final_list)

end_time = datetime.now()

print('Duration: {}'.format(end_time - start_time))


print(total_rr,total_rb,total_rg,total_bb,total_bg,total_gg)



