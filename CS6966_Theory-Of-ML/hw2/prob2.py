from __future__ import print_function, division
import matplotlib.pyplot as plt

def fx(x, eta):
    return x**2 - eta * 2 * x

def fy(y, eta):
    return y**2/4 - eta * y / 2


pos = [1,1]
eta = 0.51
x_vals = [1]
y_vals = [1]

for i in range(20):
    print(i, pos[0])
    pos[0] = fx(pos[0], eta)
    pos[1] = fy(pos[1], eta)
    x_vals.append(pos[0])
    y_vals.append(pos[1])


plt.plot(x_vals, label="x")
plt.plot(y_vals, label="y")
plt.legend(loc="best")
plt.ylim([-.1,.1])
plt.show()

