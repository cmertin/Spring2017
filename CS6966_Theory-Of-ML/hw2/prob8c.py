from __future__ import print_function, division
import numpy as np
from random import shuffle
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt


def W(pos,w,eta,itr=1):
    t = np.zeros((1,2))
    t[0, 0] = 2 * (w[0, 0] - pos[0])
    t[0, 1] = 2 * (w[0, 1] - pos[1])
    eta_ = eta**itr
    return eta_ * t


eta = .9
pos = [(1,0), (0,1), (-1,0), (0,-1)]
pos = pos * 20
x = []
y = []
points_x = [1, 0, -1, 0]
points_y = [0, 1, 0, -1]

shuffle(pos)

w = np.zeros((1,2))

for idx, p in enumerate(pos):
    x.append(w[0,0])
    y.append(w[0,1])
    #print(w[0,0], w[0,1], idx)
    w = w - W(p,w,eta,idx)

plt.scatter(points_x, points_y, color="black")
    
for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],alpha=float(i)/(len(x)-1),color="red")

plt.title("Evolution of weight vector with $\eta = 0.9^{i}$")
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.savefig("Decreasing.pdf", bbox_inches="tight")
plt.show()

eta = 1/3
w = np.zeros((1,2))
x = []
y = []

for p in pos:
    x.append(w[0,0])
    y.append(w[0,1])
    w = w - W(p,w,eta)

plt.clf()

plt.scatter(points_x, points_y, color="black")
    
for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],alpha=float(i)/(len(x)-1),color="blue")

plt.title(r"Evolution of weight vector with $\eta = \frac{1}{3}$")
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.savefig("Fixed.pdf", bbox_inches="tight")
plt.show()

