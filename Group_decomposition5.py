import pandas as pd
import numpy
from matplotlib import pyplot as plt



df_3 = pd.read_csv("recover_Jun.1_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_Jun.1_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_Jun.1_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_Jun.1_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_Jun.1_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_Jun.1_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_Jun.1_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_Jun.1_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_Jun.1_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_Jun.1_10")
rows_12 = df_12["recover_list"].values.tolist()

df_13 = pd.read_csv("recover_Jun.1_11")
rows_13 = df_13["recover_list"].values.tolist()

df_14 = pd.read_csv("recover_Jun.1_12")
rows_14 = df_14["recover_list"].values.tolist()

df_15 = pd.read_csv("recover_Jun.1_13")
rows_15 = df_15["recover_list"].values.tolist()

df_16 = pd.read_csv("recover_Jun.1_14")
rows_16 = df_16["recover_list"].values.tolist()

df_17 = pd.read_csv("recover_Jun.1_15")
rows_17 = df_17["recover_list"].values.tolist()

df_18 = pd.read_csv("recover_Jun.1_16")
rows_18 = df_18["recover_list"].values.tolist()

df_19 = pd.read_csv("recover_Jun.1_17")
rows_19 = df_19["recover_list"].values.tolist()

df_20 = pd.read_csv("recover_Jun.1_18")
rows_20 = df_20["recover_list"].values.tolist()

df_21 = pd.read_csv("recover_Jun.1_19")
rows_21 = df_21["recover_list"].values.tolist()


ys = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12,rows_13,rows_14,rows_15,rows_16,rows_17,rows_18,rows_19,rows_20,rows_21]

#ys = numpy.array(ys)
xs = [1/20,2/20,3/20,4/20,5/20,6/20,7/20,8/20,9/20,10/20,11/20,12/20,13/20,14/20,15/20,16/20,17/20,18/20,19/20]


for x, y in zip(xs, ys):
    plt.scatter([x] * len(y), y,alpha=0.5)

plt.xlabel('Pr')
plt.ylabel('R/N')
plt.title('Portion of Red Species versus Total Infected Number')


plt.violinplot(ys,xs,widths=0.05,showextrema=False,showmedians=True)
plt.show()
