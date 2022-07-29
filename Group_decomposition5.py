import pandas as pd
import numpy
from matplotlib import pyplot as plt



df_3 = pd.read_csv("recover_Jan.16_0.03lag_1")
rows_3 = df_3["recover_list"].values.tolist()

df_4 = pd.read_csv("recover_Jan.16_0.03lag_2")
rows_4 = df_4["recover_list"].values.tolist()

df_5 = pd.read_csv("recover_Jan.16_0.03lag_3")
rows_5 = df_5["recover_list"].values.tolist()

df_6 = pd.read_csv("recover_Jan.16_0.03lag_4")
rows_6 = df_6["recover_list"].values.tolist()

df_7 = pd.read_csv("recover_Jan.16_0.03lag_5")
rows_7 = df_7["recover_list"].values.tolist()

df_8 = pd.read_csv("recover_Jan.16_0.03lag_6")
rows_8 = df_8["recover_list"].values.tolist()

df_9 = pd.read_csv("recover_Jan.16_0.03lag_7")
rows_9 = df_9["recover_list"].values.tolist()

df_10 = pd.read_csv("recover_Jan.16_0.03lag_8")
rows_10 = df_10["recover_list"].values.tolist()

df_11 = pd.read_csv("recover_Jan.16_0.03lag_9")
rows_11 = df_11["recover_list"].values.tolist()

df_12 = pd.read_csv("recover_Jan.16_0.03lag_10")
rows_12 = df_12["recover_list"].values.tolist()



ys = [rows_3,rows_4,rows_5,rows_6,rows_7,rows_8,rows_9,rows_10,rows_11,rows_12]

ys = numpy.array(ys)
xs = numpy.array([1/20,2/20,3/20,4/20,5/20,6/20,7/20,8/20,9/20,10/20])

for x, y in zip(xs, ys):
    plt.scatter([x] * len(y), y,alpha=0.5)

plt.xlabel('Pr')
plt.ylabel('R/N')
plt.show()