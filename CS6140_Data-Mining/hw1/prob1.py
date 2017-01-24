from __future__ import print_function, division
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib import rc
from time import time

def BirthdayParadox(n_list, m_list):
    data = []
    for n in n_list:
        for m in m_list:
            k_list = []
            start = time()
            for m_ in range(0, m):
                hash_list = [] # Stores all the values
                rand = randint(0, n) # Generates random integer
                while rand not in hash_list:
                    hash_list.append(rand)
                    rand = randint(0, n)
                k_list.append(len(hash_list))
            k_mean = sum(k_list)/m # Calculates the mean value
            k_max = max(k_list)
            k_vals = range(0, k_max)
            k_percent = np.zeros((len(k_vals),), dtype=float)

            # Gets the total number of k's up to that given point
            for i in range(1, k_max):
                k_percent[i] = k_list.count(i) + k_percent[i-1]

            # Calculates the percentage of k's at each point
            for i in range(0, len(k_percent)):
                k_percent[i] = k_percent[i]/m

            stop = (time() - start)

            temp = [n, m, k_max, k_percent, k_mean, stop]
            data.append(temp)
    if len(m_list) == 1 and m_list[0] == 1:
        return k_list[0]
    else:
        return data
                

# Generate random numbers in the domain n
n_list = [4000]
m_list = [1]

k = BirthdayParadox(n_list, m_list)

print("k = " + str(k))

# Run m times until CheckDuplicates returns True
n_list = [4000]
m_list = [300]

data = (BirthdayParadox(n_list, m_list))[0]
percentages = data[3]
k_range = range(0, data[2])
k_mean = data[4]

print("k mean = " + str(k_mean))

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(k_range, percentages)
plt.title("Percentage of Numbers")
plt.xlabel("$k$")
plt.ylabel("$\%$")
plt.savefig("prob1_percentages.pdf", bbox_inches="tight")
#plt.show()

n_list = [4000, 10000, 100000, 250000, 1000000]
m_list = [300, 500, 1000, 5000, 10000]
plt.clf()

data = BirthdayParadox(n_list, m_list)

for m in m_list:
    n_data = []
    m_data = []
    for val in data:
        if val[1] == m:
            n_data.append(val[0])
            m_data.append(val[5])
    label = "$m = " + str(m) + "$"
    plt.plot(n_data, m_data, label=label)

plt.xlabel("$n$")
plt.ylabel("Time (seconds)")
plt.title("Run Time")
plt.legend(loc="best")
plt.xlim([min(n_list), max(n_list)])
plt.savefig("prob1_runtime.pdf", bbox_inches="tight")
#plt.show()

m_list = [300]
n_list = [4000, 10000, 50000, 100000, 500000, 1000000]
plt.clf()
data = BirthdayParadox(n_list, m_list)

for m in m_list:
    n_data = []
    m_data = []
    for val in data:
        if val[1] == m:
            n_data.append(val[0])
            m_data.append(val[5])
    label = "$m = " + str(m) + "$"
    plt.plot(n_data, m_data, label=label)

plt.xlabel("$n$")
plt.ylabel("Time (seconds)")
plt.title("Run Time $m = " + str(m_list[0]) + "$")
plt.legend(loc="best")
plt.xlim([min(n_list), max(n_list)])
plt.savefig("prob1_runtime-m300.pdf", bbox_inches="tight")
#plt.show()

