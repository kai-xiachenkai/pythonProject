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
    for i in list(group_now):
        number = random.random()
        if SIR_list[i] == "S":
            if number < beta:
                SIR_list[i] = "I"


    return SIR_list


def SIR(final_list,beta,gamma,SIR_list):

    record_SIR = SIR_list.copy()
    for i in final_list:
        for j in list(i):
            if record_SIR[j] == "I":
                SIR_list = infection(SIR_list,i,beta)

    for i in list(record_SIR):
        if record_SIR[i] == "I":
            random_number = random.random()
            if random_number < gamma:
                SIR_list[i] = "R"

    return SIR_list











