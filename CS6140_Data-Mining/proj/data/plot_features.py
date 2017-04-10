from __future__ import print_function, division
import os
import cv2
import numpy as np
import math
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib

def Read_Points(path, ext=".dat"):
    ext = "*" + ext
    x = []
    y = []
    lines = [line.strip('\n') for line in open(path)]
    for line in lines:
        xy = line.split(',')
        x.append(int(xy[0]))
        y.append(int(xy[1]))
        
    return x,y

def Read_Image(path):
    img = cv2.imread(path)
    img = np.float32(img)/255.0
    return img

img_path = sys.argv[1]
points_path = sys.argv[2]

if "Provo" in img_path:
    outfile = "Provo_Features.jpg"
else:
    outfile = "Seattle_Features.jpg"

x,y = Read_Points(points_path)
img = Read_Image(img_path)


img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w = 200
h = 200
fig = plt.figure(frameon=False)
inches = 6
w_out = inches
h_out = h/w * w_out
fig.set_size_inches(w_out, h_out, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(img2, aspect="auto")
if "Provo" in outfile:
    plt.scatter(x,y,color='r')
else:
    plt.scatter(x,y,color='b')

plt.savefig(outfile, bbox_inches="tight")
#plt.show()

cmd = "convert " + outfile + " -trim " + outfile

os.system(cmd)
