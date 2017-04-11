from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

def ReadDots(filename):
    x = []
    y = []
    lines = [line.strip() for line in open(filename)]

    for line in lines:
        temp = line.split(',')
        x.append(temp[0])
        y.append(600 - int(temp[1]))

    return x, y


city = "Provo"
filename = "_avg."
ext = ["jpg", "dat"]

dots_file = city + filename + ext[1]

x_p, y_p = ReadDots(dots_file)

city = "Seattle"
dots_file = city + filename + ext[1]

x_s, y_s = ReadDots(dots_file)

plt.scatter(x_p, y_p, c="red", label="Mormon")
plt.scatter(x_s, y_s, c="blue", label="Non-Mormon")
plt.legend(loc="center", shadow=True, fancybox=True, bbox_to_anchor = (0,0,1.65,1.75), bbox_transform=plt.gcf().transFigure)
plt.savefig("average_sift.pdf", bbox_inches="tight")
plt.show()

