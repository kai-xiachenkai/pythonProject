import random

def creat_SIR(n):
    i = 0
    SIR_list = {}
    while i < n:
        SIR_list[i] = "S"
        i = i + 1
    return SIR_list

def search_big(final_list):
    size = 0
    group_now = {}
    for i in final_list:
        size_now = len(list(i))
        if size_now > size:
            size = size_now
            group_now = i.copy()
    return group_now


def infection(SIR_list,group_now,beta):
    xr = 0
    xb = 0
    xg = 0
    for i in list(group_now):
        number = random.random()
        if SIR_list[i] == "S":
            if number < beta:
                SIR_list[i] = "I"
                if group_now[i] == "R":
                    xr = xr + 1
                elif group_now[i] == "B":
                    xb = xb + 1
                elif group_now[i] == "G":
                    xg = xg + 1




    return SIR_list,xr,xb,xg


def SIR(final_list,beta,gamma,SIR_list):

    RR = 0
    RB = 0
    RG = 0
    BR = 0
    BB = 0
    BG = 0
    GR = 0
    GB = 0
    GG = 0
    record_SIR = SIR_list.copy()
    for i in final_list:
        for j in list(i):
            if record_SIR[j] == "I":
                SIR_list,xr,xb,xg = infection(SIR_list,i,beta)
                if i[j] == "R":
                    RR = RR + xr
                    RB = RB + xb
                    RG = RG + xg
                elif i[j] == "B":
                    BR = BR + xr
                    BB = BB + xb
                    BG = BG + xg
                elif i[j] == "G":
                    GR = GR + xr
                    GB = GB + xb
                    GG = GG + xg

    for i in list(record_SIR):
        if record_SIR[i] == "I":
            random_number = random.random()
            if random_number < gamma:
                SIR_list[i] = "R"

    return SIR_list,RR,RB,RG,BR,BB,BG,GR,GB,GG











