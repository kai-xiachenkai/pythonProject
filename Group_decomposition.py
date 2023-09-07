import pandas as pd
import numpy
from matplotlib import pyplot as plt

df_1 = pd.read_csv("recover_Jun.2_1")
rows_1 = df_1["recover_list"].values.tolist()

df_2 = pd.read_csv("recover_Jun.2_2")
rows_2 = df_2["recover_list"].values.tolist()

df_3 = pd.read_csv("recover_Jun.2_3")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_Jun.2_4")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_Jun.2_5")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_Jun.2_6")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_Jun.2_7")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_Jun.2_8")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_Jun.2_9")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_Jun.2_10")
rows_10 = df_10["recover_list"].values.tolist()



ys = [rows_1,rows_2,rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10]


xs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for x, y in zip(xs, ys):
    plt.scatter([x] * len(y), y,alpha=0.3)


plt.xlabel('vf/vc')
plt.ylabel('R/N')

plt.violinplot(ys,xs,widths=0.05,showextrema=False,showmedians=True)
plt.title('vf/vc versus Number of recover node')

def EMT(n,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma):
    vc = v_rr * p_r ** 2 + v_bb * p_b ** 2 + v_gg * p_g ** 2 + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    k = (n-1)*vc/(vc + n * vf) * beta/gamma
    if k > 1:
        return 1 - 1/k
    else:
        return 0

N = 4000
n = 1000


F = 0.02
v_rr = 0.32
v_rb = 0.32*F
v_rg = 0.32*F
v_br = 0.32*F
v_bb = 0.32
v_bg = 0.32*F
v_gr = 0.32*F
v_gb = 0.32*F
v_gg = 0.32


gamma = 0.05
beta = 0.075

p_r = 0.3

p_b = 1 - p_r

p_g = 0



theory = []

for i in xs:
    v_fr = i*0.32
    v_fb = i*0.32
    v_fg = i*0.32
    theory.append(EMT(n, v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma))



plt.plot(xs,theory,label='EMT Theory')
plt.legend()
plt.show()

print(theory)

