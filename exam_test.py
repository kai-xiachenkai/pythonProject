from datetime import datetime
from EZDynamics import *
from SIR import *
from matplotlib import pyplot as plt

start_time = datetime.now()

N = 40000
n = 10000

v_fr = 0.1
v_fb = 0.1
v_fg = 0.1
v_rr = 0.2
v_rb = 0.2
v_rg = 0.2
v_br = 0.2
v_bb = 0.2
v_bg = 0.2
v_gr = 0.2
v_gb = 0.2
v_gg = 0.2

pr = 1/3

pb = 1-1/3-pr
initial_list,red_list,blue_list,green_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)

BigG = search_big(final_list)
# item = random.choice(list(BigG))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

plt.plot(groupN)
plt.show()

