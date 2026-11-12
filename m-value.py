import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *

start_time = datetime.now()

N = 4000
n = 1000

v_fr = 0.025
v_fb = 0.025
v_fg = 0.025
F = 0.5
Frg = 0.5
Frb = 0.5
Fbg = 0.5
v_rr = 0.3
v_rb = 0.3 * Frb
v_rg = 0.3 * Frg
v_br = 0.3 * Frb
v_bb = 0.3
v_bg = 0.3 * Fbg
v_gr = 0.3 * Frg
v_gb = 0.3 * Fbg
v_gg = 0.3

gamma = 0.05
beta1 = 0.01
beta3 = beta1
beta2 = beta1

pr = 1/3
pb = 1/3

initial_list, red_list, blue_list, green_list = create_initial(n, pr, pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb,
                                                  v_bg, v_gr, v_gb, v_gg)

total_run = 0
m = 1000
N1 = 1000
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



iterations = 0

SIR_list = creat_SIR(n)
BigG = search_big(final_list)


if find_R(BigG) == "NA":
    if len(red_list) == 0:
        print(0)
    else:
        item = random.choice(red_list)
        SIR_list[item] = "I"
else:
    item = find_R(BigG)
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

infection_list = []

while iterations < N1:
    groupN, iteration_runs, final_list = EZ_formation(n, m, final_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                          v_bb, v_bg, v_gr, v_gb, v_gg)
    SIR_list, rr, rb, rg, br, bb, bg, gr, gb, gg = SIR(final_list, beta1, gamma, beta2, beta3, SIR_list)
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

    infection_list.append(b/n)

print(infection_list)
plt.plot(infection_list)
plt.show()


"""
import pandas as pd


d1 = {'interaction':infection_list}
df1 = pd.DataFrame(data = d1)
df1.to_csv('infection_m100000',index=False)

"""