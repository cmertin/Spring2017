from __future__ import print_function, division
import numpy as np
import math
from random import shuffle
import itertools

def ReadFile(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    words = []
    string = None
    for line in lines:
        string = line
        for word in line.split():
            word.replace(',', '')
            word.replace('.', '')
            word.replace('!', '')
            word.replace('?', '')
            word.replace(':', '')
            word.replace(';', '')
            word.replace('\'', '')
            word.replace('\"', '')
            words.append(word)

    # Instead of using just the normal string, this reduces the white space to 1
    # character instead of multiple characters
    
    return words, string

def K_Grams_Words(words, k=2):
    grams = {}
    for i in range(len(words)):
        temp = " ".join(words[i:i+k])
        try:
            grams[temp] = grams[temp] + 1
        except:
            grams[temp] = 1
    return grams

def K_Grams_Chars(string, k=2):
    grams = {}

    for i in range(len(string)):
        temp = string[i:i+k]
        try:
            grams[temp] = grams[temp] + 1
        except:
            grams[temp] = 1
    return grams

def JacardSimilarities(file_list, data):
    js = np.zeros((len(file_list),len(file_list)), dtype=np.float64)

    for i in range(len(file_list)):
        for j in range(len(file_list)):
            left = data[i].copy()
            right = data[j].copy()
            total = 0
            sim = 0

            for key in right.keys():
                try:
                    temp = left[key]
                    sim += 1
                except:
                    pass # Do nothing

            left.update(right)
            total = len(left)
               
            js[i][j] = sim/total
    return js

def Print_KGram_Table(file_list, char2_list, char3_list, word2_list):
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{", end="")
    for i in range(4):
        if i is 0:
            print("l", end="")
        elif i is 3:
            print("r", end="")
        else:
            print("c", end="")
    print("}")
    print("\\hline\\hline")
    print("{\\bf File} & {\\bf $k_{2}$-Character} & {\\bf $k_{3}$-Character} & {\\bf $k_{2}$-Word}\\\\")
    print("\\hline")
    for i in range(len(file_list)):
        print("{\\tt " + file_list[i] + "} & ", end = "")
        print(str(len(char2_list[i])) + " & ", end = "")
        print(str(len(char3_list[i])) + " & ", end = "")
        print(str(len(word2_list[i])) + "\\\\")
    print("\\hline\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n")

def Print_JS_Table(file_list, js_matrix, caption=None):
    print("\\begin{table}[H]")
    print("\\centering")
    if caption is not None:
        print("\\caption{" + str(caption) + "}")
        
    print("\\begin{tabular}{", end="")
    for i in range(len(file_list)+1):
        if i is 0:
            print("l|", end="")
        elif i is len(file_list)-1:
            print("r", end="")
        else:
            print("c", end="")
    print("}")
    print("\\hline\\hline")
    print("& ", end="")
    for i in range(len(file_list)):
        if i is not len(file_list)-1:
            print("{\\tt " + str(file_list[i]) + "} &", end="")
        else:
            print("{\\tt " + str(file_list[i]) + "} \\\\")
    print("\\hline")
    for i in range(len(file_list)):
        print("{\\tt " + str(file_list[i]) + "} &", end="")
        for j in range(len(file_list)):
            temp = "%.3f" % js_matrix[i][j]
            if j is not len(file_list)-1:
                print(str(temp) + "& ", end="")
            else:
                print(str(temp) + "\\\\")
    print("\\hline\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n")

def KGrams(file_list):
    char2_list = []
    char3_list = []
    word2_list = []

    for file in files:
        words, string = ReadFile(file)

        grams = K_Grams_Chars(string, k=2)
        char2_list.append(grams)

        grams = K_Grams_Chars(string, k=3)
        char3_list.append(grams)

        grams = K_Grams_Words(words, k=2)
        word2_list.append(grams)

    # Prints out the table in LaTeX format for the k2-grmas and k3-gram
    Print_KGram_Table(files, char2_list, char3_list, word2_list)

    js_matrix = JacardSimilarities(files, char2_list)
    Print_JS_Table(files, js_matrix, caption="$k_{2}$-Character Jacard Similarities")

    js_matrix = JacardSimilarities(files, char3_list)
    Print_JS_Table(files, js_matrix, caption="$k_{3}$-Character Jacard Similarities")

    js_matrix = JacardSimilarities(files, word2_list)
    Print_JS_Table(files, js_matrix, caption="$k_{2}$-Word Jacard Similarities")

def MinHash(file1, file2, t, k_=3):
    min_hash = {}
    JS = 0
    words, string = ReadFile(file1)
    f1_hash = K_Grams_Chars(string, k=k_)
    
    words, string = ReadFile(file2)
    f2_hash = K_Grams_Chars(string, k=k_)

    union = []
    
    for key in f1_hash.keys():
        union.append(key)
    for key in f2_hash.keys():
        if key not in union:
            union.append(key)
    
    for item in union:
        if item in f1_hash and item in f2_hash:
            min_hash[item] = [1,1]
        elif item in f1_hash and item not in f2_hash:
            min_hash[item] = [1,0]
        else:
            min_hash[item] = [0,1]
            
    # Permute the min has by row and calculate JS
    for i in range(t):
        m1 = -1
        m2 = -1
        Xt = 0
        j = 0
        rand_list = [i for i in range(len(union))]
        shuffle(rand_list)
        
        while m1 is -1 and m2 is -1:
            j += 1
            elm = min_hash[union[rand_list[j]]]
            if elm[0] is 1:
                m1 = union[rand_list[i]]
            if elm[1] is 1:
                m2 = union[rand_list[i]]
            
        if m1 is m2:
            Xt = 1
        
        JS += (1/float(t)) * Xt
    return JS

def MinHash_FN(t_list, f1, f2):
    last_t = t_list[len(t_list)-1]
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{lr}")
    print("\\hline\\hline")
    print("$t$ & $J_{S}$\\\\")
    print("\\hline")
    for t in t_list:
        JS = MinHash(f1, f2, t)
        JS_str = "%.3f" % JS
        if t is not last_t:
           print(str(t) + " & " + JS_str + "\\\\")
        else:
            print(str(t) + " & " + JS_str + "\\\\")
            print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
        
def ProbSimilarity(s, b, r):
    return 1 - math.pow(1 - math.pow(s, b), r)
    
def ProbHash_FN(files, t, b=5.539, r=28.886):
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{lr}")
    print("\\hline\\hline")
    print("$f(D_{i}, D_{j})$ & Probability$/100$\\\\")
    print("\\hline")
    for i in range(len(files)):
        for j in range(len(files)):
            if j <= i:
                continue
            JS = MinHash(files[i], files[j], t)
            JS = ProbSimilarity(JS, b, r)
            JS_str = "%.3f" % JS
            print("$f(${\\tt " + files[i] + "}$,$ " + "{\\tt " + files[j] + "}$)$ & " + JS_str + "\\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

# Start of the main program
                  
files = ["D" + str(i) + ".txt" for i in range(1,5)]
t_list = [20, 60, 150, 300, 600]
t_list2 = [500, 700, 900, 1100, 1300, 1500, 1700, 1900]
t = 160
# KGrams(files)

#MinHash_FN(t_list, files[0], files[1])
#MinHash_FN(t_list2, files[0], files[1])

ProbHash_FN(files, t)
