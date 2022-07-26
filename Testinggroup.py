import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *

start_time = datetime.now()

N = 2000
n = 1000

v_fr = 0.03
v_fb = 0.03
v_fg = 0.03
v_rr = 0.4
v_rb = 0.1
v_rg = 0.1
v_br = 0.1
v_bb = 0.4
v_bg = 0.1
v_gr = 0.1
v_gb = 0.1
v_gg = 0.4

beta = 0.002
gamma = 0.015

pr = 0.26


pb = 1/3
initial_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)

total_run = 0
m = 1000
N1 = 1200
RR_list = []
RB_list = []
RG_list = []
BR_list = []
BB_list = []
BG_list = []
GR_list = []
GB_list = []
GG_list = []
recover_rate = []
infect_rate = []


print(0)

while total_run < 100:

    iterations = 0

    SIR_list = creat_SIR(n)
    BigG = search_big(final_list)
    item = random.choice(list(BigG))
    SIR_list[item] = "I"

    RR = 0
    RB = 0
    RG = 0
    BR = 0
    BB = 0
    BG = 0
    GR = 0
    GB = 0
    GG = 0

    while iterations < N1:
        groupN, iteration_runs, final_list = EZ_formation(n, m, final_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                          v_bb, v_bg, v_gr, v_gb, v_gg)
        SIR_list, rr, rb, rg, br, bb, bg, gr, gb, gg = SIR(final_list, beta, gamma, SIR_list)
        iterations = iterations + 1
        RR = RR + rr
        RB = RB + rb
        RG = RG + rg
        BR = BR + br
        BB = BB + bb
        BG = BG + bg
        GR = GR + gr
        GB = GB + gb
        GG = GG + gg

    a = 0
    b = 0

    for i in list(SIR_list):
        if SIR_list[i] == "R":
            a = a + 1
        if SIR_list[i] == "I":
            b = b + 1

    total_trans = RR + RB + RG + BR + BB + BG + GR + GB + GG

    recover_rate.append(a/n)
    infect_rate.append(b/n)
    RR_list.append(RR/total_trans)
    RB_list.append(RB/total_trans)
    RG_list.append(RG/total_trans)
    BR_list.append(BR/total_trans)
    BB_list.append(BB/total_trans)
    BG_list.append(BG/total_trans)
    GR_list.append(GR/total_trans)
    GB_list.append(GB/total_trans)
    GG_list.append(GG/total_trans)
    total_run = total_run + 1


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


print(RR_list)
print(RB_list)
print(RG_list)
print(BR_list)
print(BB_list)
print(BG_list)
print(GR_list)
print(GB_list)
print(GG_list)

print(recover_rate)

import pandas as pd

list = []
list.append(RR_list)
list.append(RB_list)
list.append(RG_list)
list.append(BR_list)
list.append(BB_list)
list.append(BG_list)
list.append(GR_list)
list.append(GB_list)
list.append(GG_list)

d1 = {'interaction':list}
df1 = pd.DataFrame(data = d1)
df1.to_csv('interaction_0.26',index=False)

d2 = {'recover_list':recover_rate}
df2 = pd.DataFrame(data = d2)
df2.to_csv('recover_0.26',index=False)