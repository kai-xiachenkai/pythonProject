import pandas as pd
import numpy
from matplotlib import pyplot as plt

df_1 = pd.read_csv("recover_Oct.25_0.1")
rows_1 = df_1["recover_list"].values.tolist()

df_2 = pd.read_csv("recover_Oct.25_0.15")
rows_2 = df_2["recover_list"].values.tolist()

df_3 = pd.read_csv("recover_Oct.25_0.2")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_Oct.25_0.25")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_Oct.25_0.3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_Oct.25_0.35")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_Oct.25_0.4")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_Oct.25_0.45")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_Oct.25_0.5")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_Oct.25_0.55")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_Oct.25_0.6")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_Oct.25_0.65")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_Oct.25_0.7")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_Oct.25_0.75")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_Oct.25_0.8")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_Oct.25_0.85")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_Oct.25_0.9")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_Oct.25_0.95")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_Oct.25_1")
rows_19 = df_19["recover_list"].values.tolist()


ys = [rows_1,rows_2,rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19]

xs = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

for x, y in zip(xs, ys):
    plt.scatter([x] * len(y), y,alpha=0.3)

plt.violinplot(ys,xs,widths=0.05,showextrema=False,showmedians=True)

plt.xlabel('F')
plt.ylabel('R/N')
plt.title('F versus Number of Infection with vf = 0.1')

plt.show()