import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import Eq, Symbol, solve,log
import scipy.optimize
import math
from matplotlib import cm


def EMT(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,p_r,p_b,p_g,beta,gamma):
    vc = v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2) + v_rb * p_r * p_b + v_br * p_b * p_r + v_bg * p_b * p_g + v_gb * p_g * p_b + v_rg * p_r * p_g + v_gr * p_g * p_r
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    k = vc/vf * beta/gamma
    if k < 1:
        return 0
    if k > 1:
        solution = scipy.optimize.fsolve(lambda x: -math.log(abs(1 - x)) / x - k, 0.1)
        return solution

df_3 = pd.read_csv("recover_July.2_1_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_July.2_1_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_July.2_1_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_July.2_1_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_July.2_1_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_July.2_1_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_July.2_1_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_July.2_1_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_July.2_1_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_July.2_1_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_July.2_1_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_July.2_1_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_July.2_1_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_July.2_1_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_July.2_1_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_July.2_1_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_July.2_1_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_July.2_1_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_July.2_1_19")
rows_21 = df_21["recover_list"].values.tolist()


ys = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

df_3 = pd.read_csv("recover_July.2_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_July.2_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_July.2_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_July.2_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_July.2_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_July.2_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_July.2_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_July.2_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_July.2_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_July.2_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_July.2_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_July.2_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_July.2_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_July.2_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_July.2_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_July.2_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_July.2_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_July.2_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_July.2_19")
rows_21 = df_21["recover_list"].values.tolist()


ys1 = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

df_3 = pd.read_csv("recover_July.1_2_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_July.1_2_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_July.1_2_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_July.1_2_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_July.1_2_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_July.1_2_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_July.1_2_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_July.1_2_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_July.1_2_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_July.1_2_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_July.1_2_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_July.1_2_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_July.1_2_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_July.1_2_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_July.1_2_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_July.1_2_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_July.1_2_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_July.1_2_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_July.1_2_19")
rows_21 = df_21["recover_list"].values.tolist()


ys2 = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

df_3 = pd.read_csv("recover_July.1_1_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_July.1_1_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_July.1_1_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_July.1_1_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_July.1_1_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_July.1_1_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_July.1_1_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_July.1_1_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_July.1_1_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_July.1_1_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_July.1_1_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_July.1_1_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_July.1_1_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_July.1_1_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_July.1_1_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_July.1_1_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_July.1_1_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_July.1_1_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_July.1_1_19")
rows_21 = df_21["recover_list"].values.tolist()


ys3 = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

df_3 = pd.read_csv("recover_Jun.30_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_Jun.30_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_Jun.30_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_Jun.30_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_Jun.30_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_Jun.30_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_Jun.30_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_Jun.30_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_Jun.30_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_Jun.30_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_Jun.30_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_Jun.30_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_Jun.30_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_Jun.30_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_Jun.30_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_Jun.30_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_Jun.30_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_Jun.30_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_Jun.30_19")
rows_21 = df_21["recover_list"].values.tolist()


ys4 = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]


#ys = numpy.array(ys)
xs = [1/30,2/30,3/30,4/30,5/30,6/30,7/30,8/30,9/30,10/30,11/30,12/30,13/30,14/30,15/30,16/30,17/30,18/30,19/30]

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

my_cmap = plt.get_cmap('hsv')

X1 = [xs[i] for i, data in enumerate(ys) for j in range(len(data))]
Y1 = [val for data in ys for val in data]
Z1 = [0.009 for i in range(len(Y1))]

X2 = [xs[i] for i, data in enumerate(ys1) for j in range(len(data))]
Y2 = [val for data in ys1 for val in data]
Z2 = [0.008 for i in range(len(Y2))]

X3 = [xs[i] for i, data in enumerate(ys2) for j in range(len(data))]
Y3 = [val for data in ys2 for val in data]
Z3 = [0.007 for i in range(len(Y3))]

X5 = [xs[i] for i, data in enumerate(ys3) for j in range(len(data))]
Y5 = [val for data in ys3 for val in data]
Z5 = [0.006 for i in range(len(Y5))]

X6 = [xs[i] for i, data in enumerate(ys4) for j in range(len(data))]
Y6 = [val for data in ys4 for val in data]
Z6 = [0.005 for i in range(len(Y6))]

X4 = X1 + X2 + X3 + X5 + X6
Y4 = Y1 + Y2 + Y3 + Y5 + Y6
Z4 = Z1 + Z2 + Z3 + Z5 + Z6


ax.scatter3D(X4,Y4,Z4)


N = 4000
n = 1000

v_fr = 0.025
v_fb = 0.025
v_fg = 0.025
F = 0.2
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

x_surf = np.arange(0,2/3,1/30)
z_surf = np.arange(0.005,0.01,0.001)
y_surf = np.zeros(shape=(5, 20))

x_surf, z_surf = np.meshgrid(x_surf, z_surf)

for i in range(5):
    for j in range(20):
        a = x_surf[i][j]
        b = z_surf[i][j]
        c = 1 - a - 1/3
        y_surf[i][j] = EMT(v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg,a,1/3,c,b,gamma)

print(y_surf)

ax.plot_surface(x_surf, y_surf, z_surf,alpha=0.5,color='orangered', edgecolors='yellow')




#ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot)



'''
for x, y in zip(xs, ys):
    ax.scatter3D([x] * len(y), y, 0.009, alpha=0.5,c = y,cmap = my_cmap)

for x, y in zip(xs, ys1):
    ax.scatter3D([x] * len(y), y, 0.008, alpha=0.5,c = y, cmap = my_cmap)

for x, y in zip(xs, ys2):
    ax.scatter3D([x] * len(y), y, 0.007, alpha=0.5,c = y, cmap = my_cmap)
'''

ax.set_xlabel('p_r')
ax.set_ylabel('R/N')
ax.set_zlabel('Beta')
plt.title('p_r versus Total Infected Number')

plt.show()