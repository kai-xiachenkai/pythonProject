from datetime import datetime
from EZDynamics import *
from SIR import *


n = 7
final_list = [{1:'R',2:"B",3:"G",4:"R",5:"R"},{5:"R",6:"G"}]

new_rr, new_rb,new_rg,new_bb,new_bg,new_gg = search_loop(n,final_list)

print(new_rr,new_rb,new_bb)