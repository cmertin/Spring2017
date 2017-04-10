from __future__ import print_function, division
import os
import numpy as np
import math
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib

def Read_Points(path, feat_n, ext=".dat"):
    ext = "*" + ext
    points = []
    for f in glob.glob(os.path.join(path, ext)):
        vals = []
        lines = [line.strip('\n') for line in open(f)]
        for n in feat_n:
            xy = lines[n].split(',')          
            vals.append([xy[0], xy[1]])
        points.append(vals)
    return points

provo = "Provo/"
