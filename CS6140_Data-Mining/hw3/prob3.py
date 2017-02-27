from __future__ import print_function, division
from scipy.special import gamma
import matplotlib.pyplot as plt
from math import pi

def c(d, r):
    num = pi**(d/2)/gamma(d/2 + 1)
    num = num * r**d * 2**(-d)
    num = num**(1/d)
    return r/num

def ball(d, r):
    num = pi**(d/2) * r**d
    return num/gamma(d/2 + 1)

def box(d, r):
    return (2 * r)**d

c_list = []
ball_list = []
box_list = []
r = 2
out = [2, 3, 4]

for d in range(1, 21):
    c_ = c(d,r)
    if d in out:
        c_str = "%.3f" % c_
        print("d = " + str(d) + ": " + c_str)
    c_list.append(c_)
    ball_list.append(ball(d,r))
    box_list.append(box(d,r))

plt.plot(range(1, 21), c_list)
#plt.plot(range(1, 21), ball_list)
#plt.plot(range(1, 21), box_list)
plt.xlabel("Dimension")
plt.ylabel("c")
plt.savefig("high_dimension.pdf", bbox_inches="tight")
plt.show()
