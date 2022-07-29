import random

"""
the function named, Create_initial(n), creates an initial list of length n, with each subject(R,B,G) has the same 
probability to be drawn. The initial list created is a list of individual dictionaries. For example, the list of length
4 can be: [{0:R},{1:B},{2:G},{3:R}], the number represents nth subject and the letter represents the color. 
"""

def create_initial(n,pr,pb):
    i = 0
    new_list = []
    red_list = []
    blue_list = []
    green_list = []
    while i < n:
        rnumber = random.random()
        if rnumber < pr :
            tel = {}
            tel[i] = "R"
            new_list.append(tel)
            red_list.append(i)
        elif pr < rnumber < pr+pb :
            tel = {}
            tel[i] = "B"
            new_list.append(tel)
            blue_list.append(i)
        else :
            tel = {}
            tel[i] = "G"
            new_list.append(tel)
            green_list.append(i)
        i = i+1
    return new_list,red_list,blue_list,green_list

"""
the function named, dissolve(listi), dissolves the listi that I have input into the system and stores each element
in listi in a new list. Note that this function works only for a list of dictionary. For example, with input
{1:R,2:B,3:G}, the function will return [{1:R},{2:B},{3:G}]
"""


def dissolve(listi):
    newlist = []
    for i in list(listi):
        tupple = {}
        tupple[i] = listi[i]
        newlist.append(tupple.copy())

    return newlist

"""

the function called merge_r, evaluates two listi and listj. For a identity of a specific element in listj, 
which is called identity_j, I evaluate the identity and if r-value, which is a random-drawn value,
falls in the range I define, listj will merge with listi. If r is not in the range, nothing will happen.
This function will operate when the designed elements drawn from listi is "R".

"""


def merge_r(identity_j, listi, listj, v_fr, v_rr, v_rb, v_rg, r):
    newlist = listi.copy()
    if identity_j == "R":
        if v_fr < r < v_fr + v_rr :
            newlist.update(listj)
            return newlist
    elif identity_j == "B":
        if v_fr < r < v_fr + v_rb :
            newlist.update(listj)
            return newlist
    elif identity_j == "G":
        if v_fr < r < v_fr + v_rg :
            newlist.update(listj)
            return newlist

    return "False"


"""

the function called merge_b, evaluates two listi and listj. For a identity of a specific element in listj, 
which is called identity_j, I evaluate the identity and if r-value, which is a random-drawn value,
falls in the range I define, listj will merge with listi. If r is not in the range, nothing will happen.
This function will operate when the designed elements drawn from listi is "B".

"""

def merge_b(identity_j, listi, listj, v_fb, v_br, v_bb, v_bg, r):
    newlist = listi.copy()
    if identity_j == "R":
        if v_fb < r < v_fb + v_br:
            newlist.update(listj)
            return newlist
    elif identity_j == "B":
        if v_fb < r < v_fb + v_bb:
            newlist.update(listj)
            return newlist
    elif identity_j == "G":
        if v_fb < r < v_fb + v_bg:
            newlist.update(listj)
            return newlist
    return "False"

"""

the function called mergegb, evaluates two listi and listj. For a identity of a specific element in listj, 
which is called identity_j, I evaluate the identity and if r-value, which is a random-drawn value,
falls in the range I define, listj will merge with listi. If r is not in the range, nothing will happen.
This function will operate when the designed elements drawn from listi is "G".

"""


def merge_g(identity_j, listi, listj, v_fg, v_gr, v_gb, v_gg, r):
    newlist = listi.copy()
    if identity_j == "R":
        if v_fg < r < v_fg + v_gr:
            newlist.update(listj)
            return newlist
    elif identity_j == "B":
        if v_fg < r < v_fg + v_gb:
            newlist.update(listj)
            return newlist
    elif identity_j == "G":
        if v_fg < r < v_fg + v_gg:
            newlist.update(listj)
            return newlist
    return "False"

"""
The function, called search, search for location of the index and return the index if found. 
Otherwise, it will return False.
"""

def search(list1, index):

    for i in range(len(list1)):
        m = list1[i]
        if index in list(m):
            return i

    return "False"

"""
The function, called random_number, draw a random integer from 0 to N but not a. 
"""

def random_number(a, N):
    m = random.randint(0,N-1)
    if m == a:
        return random_number(a, N)
    else:
        return m

"""
The function, called find_rbg, takes a listi and collects the number of each color in each groups and find the number of 
groups correspond to each number of color.
"""

def find_rbg(list1):
    r_number = {}
    b_number = {}
    g_number = {}
    for i in list1:
        r = 0
        b = 0
        g = 0
        for j in list(i):
            if i[j] == "R":
                r = r+1
            elif i[j] == "B":
                b = b+1
            elif i[j] == "G":
                g = g+1
        try:
            r_number[r] = r_number[r] + 1
            b_number[b] = b_number[b] + 1
            g_number[g] = g_number[g] + 1
        except KeyError:
            r_number[r] = 1
            b_number[b] = 1
            g_number[g] = 1
    return r_number, b_number, g_number

def EZ_formation(n,N,initial_list,v_fr,v_fb,v_fg,v_rr,v_rb,v_rg,v_br,v_bb,v_bg,v_gr,v_gb,v_gg):

    """
    The code iterates for N times and constantly changes the intial_list that we have put into based the EZ-group formation
    algorithm. First part of the code is to generate a random number r between 0 to 1 and select a random individual.
    This r determines the group, which this individual belongs to, will behave. After that, I will draw another individuals
    from the list and compare with the given parameters which are based on the color of the first individual.The fragmentation
    will happen when r is smaller than the set quantity, (v_fr,v_fb,v_fg). If r is bigger than the given quantity,
    then we gonna evaluate it based on the merge-function, which will either lead to a merge between two groups or nothing
    will happen.
    """
    i = 0
    groupN = []
    iteration_runs = []

    while i < N:
        r = random.random()
        random_index = random.randint(0, n - 1)  # the index of first drawn individual
        a = search(initial_list, random_index)  # the location of the group that the first one is in
        if a != "False":
            color = initial_list[a][random_index]  # color of the first individual
            state = initial_list[a]  # the group of the first individual.
            if color == "R":  # evaluate the case when first particle is red
                random_j = random_number(random_index, n)  # second individual drawn
                j_index = search(initial_list, random_j)  # the location of the group that the second one is in
                jlist = initial_list[j_index]  # the group of the second individual
                jcolor = jlist[random_j]  # the color of second individual
                if r < v_fr:  # situation where dissolve occurs.
                    m = dissolve(state)
                    initial_list.remove(state)
                    initial_list = initial_list + m
                else:
                    m = merge_r(jcolor, state, jlist, v_fr, v_rr, v_rb, v_rg, r)
                    if m != "False":  # situation where merge occurs
                        if state == jlist:
                            initial_list.remove(state.copy())
                            initial_list.append(m.copy())
                        else:
                            initial_list.remove(state.copy())
                            initial_list.remove(jlist.copy())
                            initial_list.append(m.copy())
            elif color == "B":  # evaluate the case when first particle is blue
                random_j = random_number(random_index, n)
                j_index = search(initial_list, random_j)
                jlist = initial_list[j_index]
                jcolor = jlist[random_j]
                if r < v_fb:
                    m = dissolve(state)
                    initial_list.remove(state)
                    initial_list = initial_list + m
                else:
                    m = merge_b(jcolor, state, jlist, v_fb, v_br, v_bb, v_bg, r)
                    if m != "False":
                        if state == jlist:
                            initial_list.remove(state.copy())
                            initial_list.append(m.copy())
                        else:
                            initial_list.remove(state.copy())
                            initial_list.remove(jlist.copy())
                            initial_list.append(m.copy())
            elif color == "G":  # evaluate the case when first particle is green
                random_j = random_number(random_index, n)
                j_index = search(initial_list, random_j)
                jlist = initial_list[j_index]
                jcolor = jlist[random_j]
                if r < v_fg:
                    m = dissolve(state)
                    initial_list.remove(state)
                    initial_list = initial_list + m
                else:
                    m = merge_g(jcolor, state, jlist, v_fg, v_gr, v_gb, v_gg, r)
                    if m != "False":
                        if state == jlist:
                            initial_list.remove(state.copy())
                            initial_list.append(m.copy())
                        else:
                            initial_list.remove(state.copy())
                            initial_list.remove(jlist.copy())
                            initial_list.append(m.copy())

        i = i + 1
        groupN.append(len(initial_list))
        iteration_runs.append(initial_list)

    return groupN, iteration_runs, initial_list

def average_group(N,iteration_runs):

    # calculate the number of groups versus the size of such group.

    dictionary_length = []

    for j in range(N):
        dictionary = {}
        for i in iteration_runs[j + N]:
            length = len(list(i))
            try:
                dictionary[length] = dictionary[length] + 1
            except KeyError:
                dictionary[length] = 1
        dictionary_length.append(dictionary.copy())

    numberG = {}
    numberG_s = {}

    for i in dictionary_length:
        for j in list(i):
            try:
                numberG[j] = numberG[j] + i[j]
                numberG_s[j] = numberG_s[j] + 1
            except KeyError:
                numberG[j] = i[j]
                numberG_s[j] = 1

    for i in list(numberG):
        numberG[i] = numberG[i] / numberG_s[i]

    return numberG

def search_loop(N,final_list):
    total = N*(N-1)/2
    total_rr = 0
    total_rb = 0
    total_rg = 0
    total_bb = 0
    total_bg = 0
    total_gg = 0
    for i in final_list:
        length = len(list(i))
        if length != 1:
            for m in range(length):
                for n in range(m+1,length):
                    if i[list(i)[m]] == "R" and i[list(i)[n]] == "R":
                        total_rr += 1
                    elif i[list(i)[m]] == "R" and i[list(i)[n]] == "B":
                        total_rb += 1
                    elif i[list(i)[m]] == "R" and i[list(i)[n]] == "G":
                        total_rg += 1
                    elif i[list(i)[m]] == "B" and i[list(i)[n]] == "R":
                        total_rb += 1
                    elif i[list(i)[m]] == "B" and i[list(i)[n]] == "B":
                        total_bb += 1
                    elif i[list(i)[m]] == "B" and i[list(i)[n]] == "G":
                        total_bg += 1
                    elif i[list(i)[m]] == "G" and i[list(i)[n]] == "B":
                        total_bg += 1
                    elif i[list(i)[m]] == "G" and i[list(i)[n]] == "G":
                        total_gg += 1
                    elif i[list(i)[m]] == "G" and i[list(i)[n]] == "R":
                        total_rg += 1
    total_rr = total_rr/total
    total_rb = total_rb/total
    total_rg = total_rg/total
    total_bb = total_bb/total
    total_bg = total_bg/total
    total_gg = total_gg/total
    return total_rr,total_rb,total_rg,total_bb,total_bg,total_gg



