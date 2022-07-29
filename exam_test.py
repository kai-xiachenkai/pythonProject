from datetime import datetime
from EZDynamics import *
from SIR import *

N = 2000
n = 1000

v_fr = 0.01
v_fb = 0.01
v_fg = 0.01
v_rr = 0.2
v_rb = 0.2*0.5
v_rg = 0.2*0.5
v_br = 0.2*0.5
v_bb = 0.2
v_bg = 0.2*0.5
v_gr = 0.2*0.5
v_gb = 0.2*0.5
v_gg = 0.2

pr = 2/3*1/10

pb = 1-1/3-pr
initial_list,red_list,blue_list,green_list = create_initial(n,pr,pb)

groupN, iteration_runs, final_list = EZ_formation(n, N, initial_list, v_fr, v_fb, v_fg, v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg)

BigG = search_big(final_list)
# item = random.choice(list(BigG))
if find_R(BigG) == "NA":
    if len(red_list) == 0:
        print(0)
    else:
        item = random.choice(red_list)
        print("1")
        print(item)
        print(red_list)
else:
    item = find_R(BigG)
    print("2")
    print(item)
    print(BigG)
    print(red_list)

