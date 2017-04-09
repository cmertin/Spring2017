import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

def N1(x, t=0):
    if x < t:
        return 0
    else:
        return 2*x

def N2(x, t=0):
    if x < t:
        return 0
    else:
        return 2 * (1-x) - 2*x

def N3(x, t=0):
    if x < t:
        return 0
    else:
        return - 2 * (1 - x)

def NN(x):
    n1 = N1(x)
    n2 = N2(x, t=0.5)
    n3 = N3(x, t=1)
    return n1 + n2 + n3

vals = []
x = []
#random.seed(0)
n = 500

for i in range(n):
    x.append(random.uniform(-1, 2))

for x_ in x:
    vals.append(NN(x_))

plt.scatter(x, vals)
plt.ylim([-0.05,1.05])
plt.xlim([-1,2])
plt.xlabel("x")
plt.ylabel("NN(x)")
plt.title("Single Layer Network")
plt.savefig("single_layer-13.pdf", bbox_inches="tight")
plt.show()

x = deepcopy(vals)
vals = []
for x_ in x:
    vals.append(NN(x_))

plt.clf()
plt.ylim([-0.05,1.05])
plt.xlim([-0.05,1.05])
plt.xlabel("x")
plt.ylabel("NN(NN(x))")
plt.title("Single stacked network")
plt.scatter(x, vals)
plt.savefig("stacked-13.pdf", bbox_inches="tight")
plt.show()


