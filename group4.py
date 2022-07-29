import pandas as pd

df_3 = pd.read_csv("vnew2.3")
df_4 = pd.read_csv("rb_fg")
df_5 = pd.read_csv("bb_fg")
df_6 = pd.read_csv("bg_fg")
df_1 = pd.read_csv("rr_fg")
df_2 = pd.read_csv("rg_fg")
df_7 = pd.read_csv("gg_fg")

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

plt.xlabel("fragmentation_rate of color green")
plt.ylabel("pair density")
plt.show()

plt.plot(rows_3, rows_6, '->', color='blue')
plt.plot(rows_3, rows_4, '-<', color='green')
plt.plot(rows_3, rows_2, '-s', color='red')

plt.legend(['blue_green', 'red_blue', 'red_green'])
#plt.plot(rows_3, rows_2)

plt.xlabel("fragmentation_rate of color green")
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

plt.xlabel("fragmentation_rate of color green")
plt.ylabel("pair density")
plt.show()