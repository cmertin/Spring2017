from __future__ import print_function, division
import numpy as np
from random import uniform
from random import shuffle
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt

def BuildDataSet(n,min_val=-1,max_val=1):
    dataset = []
    for i in range(n):
        dataset.append(np.asarray([uniform(min_val, max_val), uniform(min_val, max_val)]))
    return dataset

def W(pos,w,eta):
    t = np.zeros((1,2))
    t[0, 0] = 2 * (w[0, 0] - pos[0])
    t[0, 1] = 2 * (w[0, 1] - pos[1])
    return eta * t

n = 100
T_list = [10, 100, 1000]
eta = 1/2
data = BuildDataSet(n)
avg = 5


print("\\begin{table}[H]")
print("\\centering")
print("\\begin{tabular}{c c c}")
print("\\hline\\hline")
print("$T = 10$ & $T = 100$ & $T = 1,000$\\\\")
print("\\hline")
for t in T_list:
    vals = []
    for a in range(avg):
        shuffle(data)
        w = np.zeros((1,2))
        for i in range(t):
            idx = i % len(data)
            eta_ = 1/(i+1)
            w = w - W(data[idx], w, eta_)
        vals.append(np.linalg.norm(w))
    v = np.average(vals)
    v_str = "%.3f" % v
    if t != 1000:
        print(v_str + " & ", end="")
    else:
        print(v_str + "\\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")


