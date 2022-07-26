import pandas as pd
import numpy

df_3 = pd.read_csv("vnew2")
df_4 = pd.read_csv("rb_new2")
df_5 = pd.read_csv("bb_new2")
df_6 = pd.read_csv("bg_new2")
df_1 = pd.read_csv("rr_new2")
df_2 = pd.read_csv("rg_new2")
df_7 = pd.read_csv("gg_new2")

rows_1 = df_1["interaction"].values.tolist()
rows_2 = df_2["interaction"].values.tolist()
rows_3 = df_3["interaction"].values.tolist()
rows_4 = df_4["interaction"].values.tolist()
rows_5 = df_5["interaction"].values.tolist()
rows_6 = df_6["interaction"].values.tolist()
rows_7 = df_7["interaction"].values.tolist()

rows_1new = numpy.array(rows_1)
rows_2new = numpy.array(rows_2)
rows_3new = numpy.array(rows_3)
rows_4new = numpy.array(rows_4)
rows_5new = numpy.array(rows_5)
rows_6new = numpy.array(rows_6)
rows_7new = numpy.array(rows_7)

rows_1new = rows_1new*2001/3001
rows_2new = rows_2new*2001/3001
rows_4new = rows_4new*2001/3001
rows_5new = rows_5new*2001/3001
rows_6new = rows_6new*2001/3001
rows_7new = rows_7new*2001/3001

rows_8 = [sum(x) for x in zip(rows_1new,rows_2new,rows_4new,rows_5new,rows_6new,rows_7new)]

import matplotlib.pyplot as plt

#plt.plot(rows_3, rows_4)
plt.plot(rows_3new, rows_5new, '->', color='blue', label="blue_blue")
#plt.plot(rows_3, rows_6)
plt.plot(rows_3new, rows_7new, '-<', color='green', label="green_green")
plt.plot(rows_3new, rows_1new, '-s', color='red', label="red_red")
plt.legend(['blue_blue', 'green_green', 'red_red'])
#plt.plot(rows_3, rows_2)

plt.xlabel("fragmentation_rate")
plt.ylabel("pair density")
plt.show()

plt.plot(rows_3new, rows_6new, '->', color='blue')
plt.plot(rows_3new, rows_4new, '-<', color='green')
plt.plot(rows_3new, rows_2new, '-s', color='red')

plt.legend(['blue_green', 'red_blue', 'red_green'])
#plt.plot(rows_3, rows_2)

plt.xlabel("fragmentation_rate")
plt.ylabel("pair density")

plt.show()

plt.plot(rows_3new, rows_8, '-*', color='pink', label="total_P")
plt.plot(rows_3new, rows_4new,'-1', color='black', label="red_blue")
plt.plot(rows_3new, rows_5new, '->', color='blue', label="blue_blue")
plt.plot(rows_3new, rows_6new,'-2', color='orange', label="blue_green")
plt.plot(rows_3new, rows_7new, '-<', color='green', label="green_green")
plt.plot(rows_3new, rows_1new, '-s', color='red', label="red_red")
plt.plot(rows_3new, rows_2new,'-3', color='yellow', label="red_green")
plt.legend(['total_P','red_blue','blue_blue','blue_green', 'green_green', 'red_red','red_green'])

plt.xlabel("fragmentation_rate")
plt.ylabel("pair density")
plt.show()

f_list = numpy.array(rows_3)
a = (1/4+0.9/4)
y_list = a/(a+1000*f_list)

plt.plot(f_list,y_list, '-*', color='pink', label="total_P")
plt.plot(rows_3new, rows_8, '->', color='black', label="total_P")

plt.show()

print(rows_3new,rows_8)


