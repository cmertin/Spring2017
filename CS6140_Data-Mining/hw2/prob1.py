from __future__ import print_function, division
import numpy as np

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

# Start of the main program
                  
files = ["D" + str(i) + ".txt" for i in range(1,5)]

words, string = ReadFile(files[0])
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

