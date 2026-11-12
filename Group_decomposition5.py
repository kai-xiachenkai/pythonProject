import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import Eq, Symbol, solve,log
import scipy.optimize
import math

from scipy.integrate import odeint
import matplotlib.ticker as mtick





df_3 = pd.read_csv("recover_Oct.4_")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_2026.Reference2000_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_2026.Reference2000_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_2026.Reference2000_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_2026.Reference2000_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_2026.Reference2000_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_2026.Reference2000_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_2026.Reference2000_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_2026.Reference2000_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_2026.Reference2000_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_2026.Reference2000_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_2026.Reference2000_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_2026.Reference2000_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_2026.Reference2000_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_2026.Reference2000_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_2026.Reference2000_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_2026.Reference2000_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_2026.Reference2000_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_2026.Reference2000_19")
rows_21 = df_21["recover_list"].values.tolist()







#df_22 = pd.read_csv("recover_Sep.27_vf0.1_20")
#rows_22 = df_22["recover_list"].values.tolist()

#df_23 = pd.read_csv("recover_Sep.27_vf0.1_21")
#rows_23 = df_23["recover_list"].values.tolist()

ys = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]


#ys = [rows_3,rows_5,rows_7,rows_9,rows_11,rows_13,rows_15,rows_17,rows_19,rows_21]
#ys = numpy.array(ys)
xs = [1/30,2/30,3/30,4/30,5/30,6/30,7/30,8/30,9/30,10/30,11/30,12/30,13/30,14/30,15/30,16/30,17/30,18/30,19/30]

#xs = [1/30,3/30,5/30,7/30,9/30,11/30,13/30,15/30,17/30,19/30]

#xs = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

fig,ax = plt.subplots()
for x, y in zip(xs, ys):
    ax.scatter([x] * len(y), y, alpha=0.8, c ='deepskyblue')


plt.xlabel('$P_R$', size = 25)
plt.ylabel('$R/N$', size = 25)
# plt.title('Portion of Red Species versus Total Infected Number when vbb = 0.5')



vp = ax.violinplot(ys,xs,widths=0.037,showmeans=False, showextrema=True, showmedians=True)
for body in vp['bodies']:
    body.set_facecolor('red')
    body.set_edgecolor('black')
    body.set_alpha(0.3)
vp['cmaxes'].set_color('black')
vp['cmins'].set_color('black')
vp['cbars'].set_color('black')
vp['cmedians'].set_color('black')
#vp['cmeans'].set_color('black')
tmp = [np.percentile(data,[25,50,75]) for data in ys]
def get_whisker(tmp,dataset):
    whisker = []
    for quantile,data in zip(tmp,dataset):
        data = np.array(data)
        q1 = quantile[0]
        median = quantile[1]
        q3 = quantile[2]
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        upper = np.clip(upper,q3,data.max())
        lower = q1 - 1.5 * iqr
        lower = np.clip(lower,data.min(),q1)
        whisker.append((upper,lower))
    return whisker

whisker = get_whisker(tmp,ys)

ax.scatter(xs,
[quantile[1] for quantile in tmp],
marker='o',color='white',s=30,zorder=3)

ax.vlines(xs,
[quantile[0] for quantile in tmp],
[quantile[2] for quantile in tmp],
color='black',linestyle='-',lw=4)

ax.vlines(xs,
[bound[0] for bound in whisker],
[bound[1] for bound in whisker],
color='black',linestyle='-',lw=2)
#plt.boxplot(ys,positions=xs,widths=0.01)
#plt.xticks(visible=False)

def smodel(s,t,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,n):

    vc = v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2) + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    p = vc/(vc+n*vf)
    c = n - gamma/beta/p* np.log(n-1)
    dsdt = -gamma* s * np.log(s)+beta*p* s**2 - beta * p * c * s
    return dsdt

def EMT1(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,n):
    s0 = 999
    t = np.linspace(0, 1500, 1500)
    result = odeint(smodel, s0, t, args=(v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg, p_r, p_b, p_g, beta, gamma, n,))
    result1 = 1000 - result[-1]
    return result1




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
F = 0.1
Frg = 0.1
Frb = 0.1
Fbg = 0.1
v_rr = 0.3
v_rb = 0.3*Frb
v_rg = 0.3*Frg
v_br = 0.3*Frb
v_bb = 0.3
v_bg = 0.3*Fbg
v_gr = 0.3*Frg
v_gb = 0.3*Fbg
v_gg = 0.3


gamma = 0.05
beta = 0.01
beta3 = beta
beta2 = beta






theory = []

for i in xs:

    p_r = i

    p_b = 1 / 3

    p_g = 1 - p_r - p_b



    c = EMT(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma)
    if c == 0:
        print(i)
    a = EMT1(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,n)
    b = EMT1(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,n)

    if a != b:
        print(a-b)

    theory.append(a/1000)





risk = []
number = 0
for i in ys:
    individual = 0
    for j in i:
        if j > .75*theory[number]:
            individual += 1
    risk.append(individual/500)
    number = number + 1

print(risk)


plt.plot(xs,theory)
#plt.xlim(0, 0.8)
#plt.ylim(-0.05, 0.15)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.subplots_adjust(left=0.205, bottom=0.162, right=0.985, top=0.932, wspace=None, hspace=None)
#plt.axvspan(0, 1, color='green', alpha=0.1)
#plt.axvspan(1, 1, color='grey', alpha=0.2)
#plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.grid()



plt.show()


#plt.scatter(xs,risk)
#plt.show()


