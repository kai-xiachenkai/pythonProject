import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import Eq, Symbol, solve,log
import scipy.optimize
import math




df_3 = pd.read_csv("recover_July.5_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_July.5_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_July.5_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_July.5_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_July.5_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_July.5_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_July.5_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_July.5_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_July.5_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_July.5_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_July.5_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_July.5_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_July.5_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_July.5_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_July.5_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_July.5_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_July.5_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_July.5_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_July.5_19")
rows_21 = df_21["recover_list"].values.tolist()


ys = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

#ys = numpy.array(ys)
#xs = [1/30,2/30,3/30,4/30,5/30,6/30,7/30,8/30,9/30,10/30,11/30,12/30,13/30,14/30,15/30,16/30,17/30,18/30,19/30]

xs = [1/25,2/25,3/25,4/25,5/25,6/25,7/25,8/25,9/25,10/25,11/25,12/25,13/25,14/25,15/25,16/25,17/25,18/25,19/25]

#xs = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]


for x, y in zip(xs, ys):
    plt.scatter([x] * len(y), y, alpha=0.5)


plt.xlabel('Pr')
plt.ylabel('R/N')
plt.title('Pr versus Total Infected Number when pb = 1/5')


plt.violinplot(ys,xs,widths=0.05,showextrema=False,showmedians=True)



def EMT(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma):
    vc = v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2) + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    k = vc/vf * beta/gamma
    if k < 1:
        return 0
    if k > 1:
        solution = scipy.optimize.fsolve(lambda x: -math.log(abs(1 - x)) / x - k, 0.1)
        if solution > 1:
            print(k)
        return solution



N = 4000
n = 1000

v_fr = 0.025
v_fb = 0.025
v_fg = 0.025
F = 0.2
Frb = 0.2
Frg = 0.2
Fbg = 0.2
v_rg = 0.32 * Frg
v_gr = 0.32 * Frg
v_rb = 0.32 * Frb
v_br = 0.32 * Frb
v_rr = 0.32
v_bb = 0.32
v_bg = 0.32 * Fbg
v_gb = 0.32 * Fbg
v_gg = 0.32


gamma = 0.05
beta = 0.01





theory = []

for i in xs:

    p_r = i

    p_b = 1 / 5

    p_g = 1 - p_r - p_b


    a = EMT(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma)
    theory.append(a)

print(theory)

theory[1] = theory[17] = 0.6767

#theory[2] = 0.676259


#theory[4] = 0.759528
#theory[7] = 0.7
#theory[8] = 0.6853
#theory[9] = 0.674482
#theory[10] = 0.668472
#theory[11] = 0.668018
#theory[12] = 0.673502
#theory[13] = 0.684927
#theory[16] = 0.749278



risk = []
for i in ys:
    individual = 0
    for j in i:
        if j > 1/1000:
            individual += 1
    risk.append(individual/500)

print(risk)


plt.plot(xs,theory,label='EMT Theory; pb = 1/5')
plt.legend()
#plt.xlim(0, 0.8)
plt.ylim(-0.05, 0.85)

plt.show()


plt.scatter(xs,risk)
plt.show()


