import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *

start_time = datetime.now()

N = 4000
n = 1000


v_fr = 0.011
v_fb = 0.011
v_fg = 0.011
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

while v_fb < 0.026:
    initial_list,red_list,blue_list,green_list = create_initial(n,pr,pb)

    groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                      v_bb, v_bg, v_gr, v_gb, v_gg)
    iteration = 0
    total_rr, total_rb, total_rg, total_bb, total_bg, total_gg = search_loop(n, final_list)

    while iteration < 40000:
        groupN, iteration_runs, final_list = EZ_formation(n, 10, final_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                          v_bb, v_bg, v_gr, v_gb, v_gg)
        new_rr, new_rb,new_rg,new_bb,new_bg,new_gg = search_loop(n,final_list)
        total_rr = total_rr + new_rr
        total_rb = total_rb + new_rb
        total_rg = total_rg + new_rg
        total_bb = total_bb + new_bb
        total_bg = total_bg + new_bg
        total_gg = total_gg + new_gg
        iteration = iteration + 1

    vfr_list.append(v_fb)
    rr_list.append(total_rr/40001)
    rb_list.append(total_rb/40001)
    rg_list.append(total_rg/40001)
    bb_list.append(total_bb/40001)
    bg_list.append(total_bg/40001)
    gg_list.append(total_gg/40001)

    print(v_fb, v_fg, v_fr)

    v_fb = v_fb + 0.001
    #v_fr = v_fr + 0.001
    #v_fg = v_fg + 0.001

import pandas as pd

d1 = {'interaction':vfr_list}
df1 = pd.DataFrame(data = d1)
df1.to_csv('Jun21_vfb2',index=False)

d2 = {'interaction':rr_list}
df2 = pd.DataFrame(data = d2)
df2.to_csv('Jun21_rr_vfb2',index=False)

d3 = {'interaction':rb_list}
df3 = pd.DataFrame(data = d3)
df3.to_csv('Jun21_rb_vfb2',index=False)

d4 = {'interaction':rg_list}
df4 = pd.DataFrame(data = d4)
df4.to_csv('Jun21_rg_vfb2',index=False)

d5 = {'interaction':bb_list}
df5 = pd.DataFrame(data = d5)
df5.to_csv('Jun21_bb_vfb2',index=False)

d6 = {'interaction':bg_list}
df6 = pd.DataFrame(data = d6)
df6.to_csv('Jun21_bg_vfb2',index=False)

d7 = {'interaction':gg_list}
df7 = pd.DataFrame(data = d7)
df7.to_csv('Jun21_gg_vfb2',index=False)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
