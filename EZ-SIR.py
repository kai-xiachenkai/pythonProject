import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from EZDynamics import *
from SIR import *

start_time = datetime.now()

N = 20000
n = 10000


v_fr = 0.025
v_fb = 0.025
v_fg = 1/5
v_rr = 0.32
v_rb = 0.2*0.32
v_rg = 1/3
v_br = 0.2*0.32
v_bb = 0.32
v_bg = 1/3
v_gr = 1/3
v_gb = 1/3
v_gg = 1/3

beta = 0.005
gamma = 0.05


pr = 0.05

recover_rate = []
pr_list = []

while pr < 0.1:
    pb = 1.01 - pr

    initial_list = create_initial(n, pr, pb)

    groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                      v_bb, v_bg, v_gr, v_gb, v_gg)

    N1 = 100
    iterations = 0
    m = 5000

    SIR_list = creat_SIR(n)
    BigG = search_big(final_list)
    check = 0
    for i in list(BigG):
        if BigG[i] == "R":
            check = 1
            SIR_list[i] = "I"
            print(0)
            break

    if check == 0:
        item = random.choice(list(BigG))
        SIR_list[item] = "I"
        print(1)



    # infect_list = []
    # recover_list = []
    # iteration_list = []

    while iterations < N1:
        groupN, iteration_runs, final_list = EZ_formation(n, m, final_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br,
                                                          v_bb, v_bg, v_gr, v_gb, v_gg)
        SIR_list = SIR(final_list, beta, gamma, SIR_list)
        iterations = iterations + 1

    a = 0
    b = 0

    for i in list(SIR_list):
        if SIR_list[i] == "R":
            a = a + 1
        if SIR_list[i] == "I":
            b = b + 1
    recover_rate.append(a/n)
    pr_list.append(pr)

    pr = pr + 0.05

plt.scatter(pr_list, recover_rate, c='red', marker='o')
plt.show()


"""    
pb = 1 - pr
initial_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)


N1 = 100
iterations = 0
m = 2000

SIR_list = creat_SIR(n)
BigG = search_big(final_list)
item = random.choice(list(BigG))
SIR_list[item] = "I"

#infect_list = []
#recover_list = []
#iteration_list = []


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
    
"""


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

"""
plt.scatter(iteration_list,infect_list, c='red', marker='o')
plt.scatter(iteration_list,recover_list, c='green', marker='v')

plt.legend(["infection_rate", "recover_rate"])
plt.xlabel("iterations")
plt.ylabel("ratio")

plt.show()

"""