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
v_rr = 0.5
v_rb = 0.5*0.5
v_rg = 0.5*0.5
v_br = 0.5*0.5
v_bb = 0.5
v_bg = 0.5*0.5
v_gr = 0.5*0.5
v_gb = 0.5*0.5
v_gg = 0.5


gamma = 0.015
beta2 = 0.002
beta3 = 0.3*beta2
beta1 = beta2*0.3

pr = 9/20

pb = 1/2*10/10

initial_list,red_list,blue_list,green_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)

total_run = 0
m = 500
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
trans_rate = []
recover_rate = []
infect_rate = []


print(0)

while total_run < 100:

    iterations = 0

    SIR_list = creat_SIR(n)
    BigG = search_big(final_list)
    #item = random.choice(list(BigG))
    if find_B(BigG) == "NA":
        if len(blue_list) == 0:
            print(0)
        else:
            item = random.choice(blue_list)
            SIR_list[item] = "I"
    else:
        item = find_B(BigG)
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
        SIR_list, rr, rb, rg, br, bb, bg, gr, gb, gg = SIR(final_list, beta1, gamma,beta2,beta3, SIR_list)
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


    RR_list.append(RR)
    RB_list.append(RB)
    RG_list.append(RG)
    BR_list.append(BR)
    BB_list.append(BB)
    BG_list.append(BG)
    GR_list.append(GR)
    GB_list.append(GB)
    GG_list.append(GG)
    trans_rate.append(total_trans)
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
list.append(trans_rate)

d1 = {'interaction':list}
df1 = pd.DataFrame(data = d1)
df1.to_csv('Infection_Jan.17_0.03_IM4_10',index=False)

d2 = {'recover_list':recover_rate}
df2 = pd.DataFrame(data = d2)
df2.to_csv('recover_Jan.17_0.03_IM4_10',index=False)
