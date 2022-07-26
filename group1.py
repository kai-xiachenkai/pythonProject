import pandas as pd
import numpy

df_3 = pd.read_csv("vfb")
df_4 = pd.read_csv("rb_fb")
df_5 = pd.read_csv("bb_fb")
df_6 = pd.read_csv("bg_fb")
df_1 = pd.read_csv("rr_fb")
df_2 = pd.read_csv("rg_fb")
df_7 = pd.read_csv("gg_fb")

rows_1 = df_1["interaction"].values.tolist()
rows_2 = df_2["interaction"].values.tolist()
rows_3 = df_3["interaction"].values.tolist()
rows_4 = df_4["interaction"].values.tolist()
rows_5 = df_5["interaction"].values.tolist()
rows_6 = df_6["interaction"].values.tolist()
rows_7 = df_7["interaction"].values.tolist()

rows_8 = [sum(x) for x in zip(rows_1,rows_2,rows_4,rows_5,rows_6,rows_7)]

import matplotlib.pyplot as plt

#plt.plot(rows_3, rows_4)
plt.plot(rows_3, rows_5, '->', color='blue', label="blue_blue")
#plt.plot(rows_3, rows_6)
plt.plot(rows_3, rows_7, '-<', color='green', label="green_green")
plt.plot(rows_3, rows_1, '-s', color='red', label="red_red")
plt.legend(['blue_blue', 'green_green', 'red_red'])
#plt.plot(rows_3, rows_2)

plt.xlabel("fragmentation_rate of color blue")
plt.ylabel("pair density")
plt.show()

plt.plot(rows_3, rows_6, '->', color='blue')
plt.plot(rows_3, rows_4, '-<', color='green')
plt.plot(rows_3, rows_2, '-s', color='red')

plt.legend(['blue_green', 'red_blue', 'red_green'])
#plt.plot(rows_3, rows_2)

plt.xlabel("fragmentation_rate of color blue")
plt.ylabel("pair density")

plt.show()

plt.plot(rows_3, rows_8, '-*', color='pink', label="total_P")
plt.plot(rows_3, rows_4,'-1', color='black', label="red_blue")
plt.plot(rows_3, rows_5, '->', color='blue', label="blue_blue")
plt.plot(rows_3, rows_6,'-2', color='orange', label="blue_green")
plt.plot(rows_3, rows_7, '-<', color='green', label="green_green")
plt.plot(rows_3, rows_1, '-s', color='red', label="red_red")
plt.plot(rows_3, rows_2,'-3', color='yellow', label="red_green")
plt.legend(['total_P','red_blue','blue_blue','blue_green', 'green_green', 'red_red','red_green'])

plt.xlabel("fragmentation_rate of color blue")
plt.ylabel("pair density")
plt.show()

f_list = numpy.array(rows_3)
a = 0.5*(1/3+0.6)
y_list = a/(a+10000*(0.05/3+1/3*f_list))

plt.plot(f_list,y_list, '-*', color='pink', label="total_P")
plt.plot(rows_3, rows_8, '->', color='black', label="total_P")

plt.show()