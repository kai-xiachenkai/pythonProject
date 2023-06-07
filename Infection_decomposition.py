import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean

df_3 = pd.read_csv("infection_Jan.19_0.03_IM2_1")
rows_3 = df_3["interaction"].values.tolist()

df_4 = pd.read_csv("infection_Jan.19_0.03_IM2_2")
rows_4 = df_4["interaction"].values.tolist()

df_5 = pd.read_csv("infection_Jan.19_0.03_IM2_3")
rows_5 = df_5["interaction"].values.tolist()

df_6 = pd.read_csv("infection_Jan.19_0.03_IM2_4")
rows_6 = df_6["interaction"].values.tolist()

df_7 = pd.read_csv("infection_Jan.19_0.03_IM2_5")
rows_7 = df_7["interaction"].values.tolist()

df_8 = pd.read_csv("infection_Jan.19_0.03_IM2_6")
rows_8 = df_8["interaction"].values.tolist()

df_9 = pd.read_csv("infection_Jan.19_0.03_IM2_7")
rows_9 = df_9["interaction"].values.tolist()

df_10 = pd.read_csv("infection_Jan.19_0.03_IM2_8")
rows_10 = df_10["interaction"].values.tolist()

df_11 = pd.read_csv("infection_Jan.19_0.03_IM2_9")
rows_11 = df_11["interaction"].values.tolist()

df_12 = pd.read_csv("infection_Jan.19_0.03_IM2_10")
rows_12 = df_12["interaction"].values.tolist()


rr_1 = list(rows_3[0][1:-1].split(", "))
rb_1 = list(rows_3[1][1:-1].split(", "))
rg_1 = list(rows_3[2][1:-1].split(", "))
br_1 = list(rows_3[3][1:-1].split(", "))
bb_1 = list(rows_3[4][1:-1].split(", "))
bg_1 = list(rows_3[5][1:-1].split(", "))
gr_1 = list(rows_3[6][1:-1].split(", "))
gb_1 = list(rows_3[7][1:-1].split(", "))
gg_1 = list(rows_3[8][1:-1].split(", "))
trans1 = list(rows_3[9][1:-1].split(", "))

rr_1 = [int(i) for i in rr_1]
rb_1 = [int(i) for i in rb_1]
rg_1 = [int(i) for i in rg_1]
br_1 = [int(i) for i in br_1]
bb_1 = [int(i) for i in bb_1]
bg_1 = [int(i) for i in bg_1]
gr_1 = [int(i) for i in gr_1]
gb_1 = [int(i) for i in gb_1]
gg_1 = [int(i) for i in gg_1]
trans1 = [int(i) for i in trans1]



rr_list1 = []
rb_list1 = []
rg_list1 = []
br_list1 = []
bb_list1 = []
bg_list1 = []
gr_list1 = []
gb_list1 = []
gg_list1 = []

for i in range(100):
    if trans1[i] == 0:
        rr_list1.append(0)
        rb_list1.append(0)
        rg_list1.append(0)
        br_list1.append(0)
        bb_list1.append(0)
        bg_list1.append(0)
        gr_list1.append(0)
        gb_list1.append(0)
        gg_list1.append(0)
    else:
        rr_list1.append(rr_1[i]/trans1[i])
        rb_list1.append(rb_1[i]/trans1[i])
        rg_list1.append(rg_1[i]/trans1[i])
        br_list1.append(br_1[i]/trans1[i])
        bb_list1.append(bb_1[i]/trans1[i])
        bg_list1.append(bg_1[i]/trans1[i])
        gr_list1.append(gr_1[i]/trans1[i])
        gb_list1.append(gb_1[i]/trans1[i])
        gg_list1.append(gg_1[i]/trans1[i])

print(rr_list1)



