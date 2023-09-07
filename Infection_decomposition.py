import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean

def extractlist(rows):
    rr_1 = list(rows[0][1:-1].split(", "))
    rb_1 = list(rows[1][1:-1].split(", "))
    rg_1 = list(rows[2][1:-1].split(", "))
    br_1 = list(rows[3][1:-1].split(", "))
    bb_1 = list(rows[4][1:-1].split(", "))
    bg_1 = list(rows[5][1:-1].split(", "))
    gr_1 = list(rows[6][1:-1].split(", "))
    gb_1 = list(rows[7][1:-1].split(", "))
    gg_1 = list(rows[8][1:-1].split(", "))
    trans1 = list(rows[9][1:-1].split(", "))

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
            rr_list1.append(rr_1[i] / trans1[i])
            rb_list1.append(rb_1[i] / trans1[i])
            rg_list1.append(rg_1[i] / trans1[i])
            br_list1.append(br_1[i] / trans1[i])
            bb_list1.append(bb_1[i] / trans1[i])
            bg_list1.append(bg_1[i] / trans1[i])
            gr_list1.append(gr_1[i] / trans1[i])
            gb_list1.append(gb_1[i] / trans1[i])
            gg_list1.append(gg_1[i] / trans1[i])

    return rr_list1,rb_list1,rg_list1,br_list1,bb_list1,bg_list1,gr_list1,gb_list1,gg_list1




df_3 = pd.read_csv("Infection_July.12_1")
rows_3 = df_3["interaction"].values.tolist()

df_4 = pd.read_csv("Infection_July.12_2")
rows_4 = df_4["interaction"].values.tolist()

df_5 = pd.read_csv("Infection_July.12_3")
rows_5 = df_5["interaction"].values.tolist()

df_6 = pd.read_csv("Infection_July.12_4")
rows_6 = df_6["interaction"].values.tolist()

df_7 = pd.read_csv("Infection_July.12_5")
rows_7 = df_7["interaction"].values.tolist()

df_8 = pd.read_csv("Infection_July.12_6")
rows_8 = df_8["interaction"].values.tolist()

df_9 = pd.read_csv("Infection_July.12_7")
rows_9 = df_9["interaction"].values.tolist()

df_10 = pd.read_csv("Infection_July.12_8")
rows_10 = df_10["interaction"].values.tolist()

df_11 = pd.read_csv("Infection_July.12_9")
rows_11 = df_11["interaction"].values.tolist()

df_12 = pd.read_csv("Infection_July.12_10")
rows_12 = df_12["interaction"].values.tolist()

df_13 = pd.read_csv("Infection_July.12_11")
rows_13 = df_13["interaction"].values.tolist()

df_14 = pd.read_csv("Infection_July.12_12")
rows_14 = df_14["interaction"].values.tolist()

df_15 = pd.read_csv("Infection_July.12_13")
rows_15 = df_15["interaction"].values.tolist()

df_16 = pd.read_csv("Infection_July.12_14")
rows_16 = df_16["interaction"].values.tolist()

df_17 = pd.read_csv("Infection_July.12_15")
rows_17 = df_17["interaction"].values.tolist()

df_18 = pd.read_csv("Infection_July.12_16")
rows_18 = df_18["interaction"].values.tolist()

df_19 = pd.read_csv("Infection_July.12_17")
rows_19 = df_19["interaction"].values.tolist()

df_20 = pd.read_csv("Infection_July.12_18")
rows_20 = df_20["interaction"].values.tolist()

df_21 = pd.read_csv("Infection_July.12_19")
rows_21 = df_21["interaction"].values.tolist()





rr_list1,rb_list1,rg_list1,br_list1,bb_list1,bg_list1,gr_list1,gb_list1,gg_list1 = extractlist(rows_3)
rr_list2,rb_list2,rg_list2,br_list2,bb_list2,bg_list2,gr_list2,gb_list2,gg_list2 = extractlist(rows_4)
rr_list3,rb_list3,rg_list3,br_list3,bb_list3,bg_list3,gr_list3,gb_list3,gg_list3 = extractlist(rows_5)
rr_list4,rb_list4,rg_list4,br_list4,bb_list4,bg_list4,gr_list4,gb_list4,gg_list4 = extractlist(rows_6)
rr_list5,rb_list5,rg_list5,br_list5,bb_list5,bg_list5,gr_list5,gb_list5,gg_list5 = extractlist(rows_7)
rr_list6,rb_list6,rg_list6,br_list6,bb_list6,bg_list6,gr_list6,gb_list6,gg_list6 = extractlist(rows_8)
rr_list7,rb_list7,rg_list7,br_list7,bb_list7,bg_list7,gr_list7,gb_list7,gg_list7 = extractlist(rows_9)
rr_list8,rb_list8,rg_list8,br_list8,bb_list8,bg_list8,gr_list8,gb_list8,gg_list8 = extractlist(rows_10)
rr_list9,rb_list9,rg_list9,br_list9,bb_list9,bg_list9,gr_list9,gb_list9,gg_list9 = extractlist(rows_11)
rr_list10,rb_list10,rg_list10,br_list10,bb_list10,bg_list10,gr_list10,gb_list10,gg_list10 = extractlist(rows_12)
rr_list11,rb_list11,rg_list11,br_list11,bb_list11,bg_list11,gr_list11,gb_list11,gg_list11 = extractlist(rows_13)
rr_list12,rb_list12,rg_list12,br_list12,bb_list12,bg_list12,gr_list12,gb_list12,gg_list12 = extractlist(rows_14)
rr_list13,rb_list13,rg_list13,br_list13,bb_list13,bg_list13,gr_list13,gb_list13,gg_list13 = extractlist(rows_15)
rr_list14,rb_list14,rg_list14,br_list14,bb_list14,bg_list14,gr_list14,gb_list14,gg_list14 = extractlist(rows_16)
rr_list15,rb_list15,rg_list15,br_list15,bb_list15,bg_list15,gr_list15,gb_list15,gg_list15 = extractlist(rows_17)
rr_list16,rb_list16,rg_list16,br_list16,bb_list16,bg_list16,gr_list16,gb_list16,gg_list16 = extractlist(rows_18)
rr_list17,rb_list17,rg_list17,br_list17,bb_list17,bg_list17,gr_list17,gb_list17,gg_list17 = extractlist(rows_19)
rr_list18,rb_list18,rg_list18,br_list18,bb_list18,bg_list18,gr_list18,gb_list18,gg_list18 = extractlist(rows_20)
rr_list19,rb_list19,rg_list19,br_list19,bb_list19,bg_list19,gr_list19,gb_list19,gg_list19 = extractlist(rows_21)

rr_list = [rr_list1,rr_list2,rr_list3,rr_list4,rr_list5,rr_list6,rr_list7,rr_list8,rr_list9,rr_list10,rr_list11,rr_list12,rr_list13,rr_list14,rr_list15,rr_list16,rr_list17,rr_list18,rr_list19]
rb_list = [rb_list1,rb_list2,rb_list3,rb_list4,rb_list5,rb_list6,rb_list7,rb_list8,rb_list9,rb_list10,rb_list11,rb_list12,rb_list13,rb_list14,rb_list15,rb_list16,rb_list17,rb_list18,rb_list19]
rg_list = [rg_list1,rg_list2,rg_list3,rg_list4,rg_list5,rg_list6,rg_list7,rg_list8,rg_list9,rg_list10,rg_list11,rg_list12,rg_list13,rg_list14,rg_list15,rg_list16,rg_list17,rg_list18,rg_list19]
br_list = [br_list1,br_list2,br_list3,br_list4,br_list5,br_list6,br_list7,br_list8,br_list9,br_list10,br_list11,br_list12,br_list13,br_list14,br_list15,br_list16,br_list17,br_list18,br_list19]
bb_list = [bb_list1,bb_list2,bb_list3,bb_list4,bb_list5,bb_list6,bb_list7,bb_list8,bb_list9,bb_list10,bb_list11,bb_list12,bb_list13,bb_list14,bb_list15,bb_list16,bb_list17,bb_list18,bb_list19]
bg_list = [bg_list1,bg_list2,bg_list3,bg_list4,bg_list5,bg_list6,bg_list7,bg_list8,bg_list9,bg_list10,bg_list11,bg_list12,bg_list13,bg_list14,bg_list15,bg_list16,bg_list17,bg_list18,bg_list19]
gr_list = [gr_list1,gr_list2,gr_list3,gr_list4,gr_list5,gr_list6,gr_list7,gr_list8,gr_list9,gr_list10,gr_list11,gr_list12,gr_list13,gr_list14,gr_list15,gr_list16,gr_list17,gr_list18,gr_list19]
gb_list = [gb_list1,gb_list2,gb_list3,gb_list4,gb_list5,gb_list6,gb_list7,gb_list8,gb_list9,gb_list10,gb_list11,gb_list12,gb_list13,gb_list14,gb_list15,gb_list16,gb_list17,gb_list18,gb_list19]
gg_list = [gg_list1,gg_list2,gg_list3,gg_list4,gg_list5,gg_list6,gg_list7,gg_list8,gg_list9,gg_list10,gg_list11,gg_list12,gg_list13,gg_list14,gg_list15,gg_list16,gg_list17,gg_list18,gg_list19]


xs = [1/30,2/30,3/30,4/30,5/30,6/30,7/30,8/30,9/30,10/30,11/30,12/30,13/30,14/30,15/30,16/30,17/30,18/30,19/30]

for x, y in zip(xs, rb_list):
    plt.scatter([x] * len(y), y, alpha=0.5)


plt.xlabel('Pr')
plt.ylabel('Spread from R to R')
plt.title('Chance of Spreading from R to R during the SIR')


plt.violinplot(rr_list,xs,widths=0.05,showextrema=False,showmedians=True)

plt.show()


for x, y in zip(xs, gb_list):
    plt.scatter([x] * len(y), y, alpha=0.5)


plt.xlabel('Pr')
plt.ylabel('Spread from G to B')
plt.title('Chance of Spreading from G to B during the SIR')


plt.violinplot(gb_list,xs,widths=0.05,showextrema=False,showmedians=True)


plt.show()