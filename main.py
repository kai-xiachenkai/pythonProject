import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *

start_time = datetime.now()

N = 20000
n = 10000

v_fr = 0.03
v_fb = 0.03
v_fg = 1/5
v_rr = 0.5
v_rb = 0.1*0.5
v_rg = 1 / 3
v_br = 0.1*0.5
v_bb = 0.5
v_bg = 1 / 3
v_gr = 1 / 3
v_gb = 1 / 3
v_gg = 1 / 3

beta = 0.002
gamma = 0.015

pr = 0.5


pb = 1 - pr
initial_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)


N1 = 1000
iterations = 0
m = 10000

SIR_list = creat_SIR(n)
BigG = search_big(final_list)
item = random.choice(list(BigG))
SIR_list[item] = "I"

infect_list = []
recover_list = []
iteration_list = []

print(0)

while iterations < N1:
    groupN, iteration_runs, final_list = EZ_formation(n, m, final_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)
    SIR_list = SIR(final_list, beta, gamma, SIR_list)
    iterations = iterations + 1


    a = 0
    b = 0

    for i in list(SIR_list):
        if SIR_list[i] == "R":
            a = a + 1
        if SIR_list[i] == "I":
            b = b + 1
    iteration_list.append(iterations)
    infect_list.append(b/n)
    recover_list.append(a/n)



end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


plt.scatter(iteration_list,infect_list, c='red', marker='o')
plt.scatter(iteration_list,recover_list, c='green', marker='v')

plt.legend(["infection_rate", "recover_rate"])
plt.xlabel("iterations")
plt.ylabel("ratio")

plt.show()

