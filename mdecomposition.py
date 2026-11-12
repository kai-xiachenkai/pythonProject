import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import Eq, Symbol, solve,log
import scipy.optimize
import math
from scipy.integrate import odeint






def smodel(s,t,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,N):

    vc = v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2) + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    p = vc/(vc+N*vf)
    c = N - gamma/beta/p* np.log(N-1)
    dsdt = -gamma* s * np.log(s)+beta*p* s**2 - beta * p * c * s
    return dsdt

N = 1000

p_r = 1/3
p_b = 1/3
p_g = 1 - p_r - p_b

v_fr = 0.025
v_fb = 0.025
v_fg = 0.025
F = 1
Frb = 1
Frg = 1
Fbg = 1
v_rg = 0.3 * Frg
v_gr = 0.3 * Frg
v_rb = 0.3 * Frb
v_br = 0.3 * Frb
v_rr = 0.3
v_bb = 0.3
v_bg = 0.3 * Fbg
v_gb = 0.3 * Fbg
v_gg = 0.3

gamma = 0.05
beta = 0.025


S = 1
s0 = 999
t = np.linspace(0,1200,1200)
result = odeint(smodel,s0,t,args=(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,N,))
max = result[-1]
print(1000 - max)



def imodel(s,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,N):

    vc = v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2) + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    p = vc/(vc+N*vf)
    c = N - gamma / beta / p * np.log(N - 1)

    I = gamma/beta/p * np.log(s) - s + c

    return I


result2 = imodel(result,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma,N)

"""
plt.plot(t,result2/1000,label='EMT theory')
plt.show()


"""

df_3 = pd.read_csv("infection_m1")
rows_3 = df_3["interaction"].values.tolist()

df_4 = pd.read_csv("infection_m10")
rows_4 = df_4["interaction"].values.tolist()

df_5 = pd.read_csv("infection_m100")
rows_5 = df_5["interaction"].values.tolist()

df_6 = pd.read_csv("infection_m1000")
rows_6 = df_6["interaction"].values.tolist()

df_7 = pd.read_csv("infection_m10000")
rows_7 = df_6["interaction"].values.tolist()

df_8 = pd.read_csv("infection_m100000")
rows_8 = df_6["interaction"].values.tolist()

plt.plot(t,result2/1000,label='EMT theory')
plt.scatter(t,rows_3,label='$m=1$',marker='d',s = 25,alpha=0.8)
plt.scatter(t,rows_4,label='$m=10$',marker='+',s = 25,alpha=0.8)
plt.scatter(t,rows_5,label='$m=100$',marker='v',s = 25,alpha=0.8)
plt.scatter(t,rows_6,label='$m=1000$',marker = '<',s = 25,alpha=0.8)
plt.scatter(t,rows_7,label='$m=10000$',marker = '>',s = 25,alpha=0.8)
plt.scatter(t,rows_8,label='$m=100000$',marker = 'x',s = 25,alpha=0.5,color = "pink")
plt.xlim(0, 400)
plt.ylabel('$I/N$',size = 20)
plt.xlabel("timestep",size = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.grid()
plt.subplots_adjust(left=0.14, bottom=0.162, right=0.92, top=0.932, wspace=None, hspace=None)
leg = plt.legend(loc='upper right')
plt.show()
