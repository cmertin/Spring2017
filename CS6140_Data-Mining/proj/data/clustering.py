from __future__ import print_function, division
import os
import numpy as np
import math
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import sklearn.cluster
from numpy.linalg import svd
import itertools
import sklearn
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import complete, dendrogram


def Read_FeatureIDs(path):
    points = []
    lines = [line.strip('\n') for line in open(path)]
    for line in lines:
        points.append(int(line))
    return points

def Read_Points(path, feat_n, pts=None, ext=".dat"):
    ext = "*" + ext
    files = []
    if pts == None:
        points = []
    else:
        points = pts
    for f in glob.glob(os.path.join(path, ext)):
        files.append(f)
        vals = []
        lines = [line.strip('\n') for line in open(f)]
        '''
        for line in lines:
            xy = line.split(',')
            vals.append(int(xy[1]))
            #vals.extend([int(xy[0]), int(xy[1])])
        '''
        for n in feat_n:
            xy = lines[n].split(',')
            vals.append(int(xy[1]))
            #vals.extend([int(xy[0]), int(xy[1])])
            #vals.append([int(xy[0]), int(xy[1])])
        
        points.append(vals)
    return points, files

provo = "Provo_Affine/"
seattle = "Seattle_Affine/"
f_ids = "Feature_Numbers.dat"

f_ids = Read_FeatureIDs(f_ids)
provo, provo_f = Read_Points(provo, f_ids)
seattle, seattle_f = Read_Points(seattle, f_ids)

places = ["Seattle, WA", "Miami, FL", "Los Angeles, CA", "New York, NY", "Denver, CO", "Dallas, TX", "Las Vegas, NV", "Raleigh, NC"]

p_labels = ["Provo, UT"] * len(provo)

for i in range(len(seattle)):
    idx = randint(0, len(places)-1)
    p_labels.append(places[idx])

points = np.asarray(provo + seattle)

labels = ([0] * len(provo)) + ([1] * len(seattle))
files = provo_f + seattle_f

# Affinity = {"euclidean", "manhattan", "cosine"}
# Linkage = {"complete", "average"}
# eu com
Hclustering = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
Hclustering.fit(points)

#print(Hclustering.labels_)

lab = list(Hclustering.labels_)
#print(type(lab))

count = 0
total = 0
for idx, lbl in enumerate(lab):
    if lbl == labels[idx]:
        count += 1
    total += 1

print(str(count) + "/" + str(total), "%.3f" % float(count/total))


dist = 1 - euclidean_distances(points)
linkage_matrix = complete(dist)

plt.clf()
fig, ax = plt.subplots(figsize=(10,30))
ax = dendrogram(linkage_matrix, orientation="left", labels=p_labels)

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.tight_layout()
plt.savefig("dendrogram.pdf", bbox_inches="tight", dpi=200)
#plt.show()

#print(seattle_f)
vals = []
k = 1000
for i in range(k):
    print(str(i+1) + '/' + str(k))
    clustering = KMeans(n_clusters=2, n_init=1, tol=0.01, algorithm="full", precompute_distances=True, copy_x=True, init="random")#, random_state=0)
    clustering.fit(points)
    #print(clustering.labels_)

    count = 0
    total = 0
    for idx, lbl in enumerate(list(clustering.labels_)):
        if lbl == labels[idx]:
            count += 1
        total += 1
    vals.append(float(count/total))

plt.clf()
fig, ax = plt.subplots()
vals = np.sort(vals)
vals2 = vals/(np.asarray(vals)*0.001).sum()
vals2 = np.cumsum(vals2 * 0.001)
plt.plot(vals, vals2)
plt.xlabel("Labeling Accuracy")
plt.ylabel("Percent Occurance")
plt.title("CDF of KMeans with " + str(k) + " Trials")
plt.savefig("kmeans_cdf.pdf", bbox_inches="tight")
#plt.show()


#print(clustering.cluster_centers_)

print(str(count) + "/" + str(total), "%.3f" % float(count/total))

U,s,V = svd(points)

colors = []

for idx, lbl in enumerate(list(clustering.labels_)):
    if lbl == 0:
        colors.append('b')
    else:
        colors.append('r')

colors2 = []
for idx, lbl in enumerate(list(labels)):
    if lbl == 0:
        colors2.append('b')
    else:
        colors2.append('r')

colors3 = []
for idx, lbl in enumerate(list(labels)):
    if lbl == clustering.labels_[idx]:
        colors3.append('g')
    else:
        colors3.append('r')

S = np.diag(s)
shape = (np.shape(U)[1], np.shape(S)[1])
S.resize(shape)
Ux = np.dot(U,S)

x1 = []
x2 = []

dim = range(1)

     
plt.clf()

for idx, lbl in enumerate(clustering.labels_):
    if lbl == 0:#clustering.labels_[idx]:
        x1.append(Ux[idx,0])
    else:
        x2.append(Ux[idx,0])

f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(x1, bins=20, color='lightblue', alpha=0.5, label="Mormon")
axarr[0].hist(x2, bins=20, color='salmon', alpha=0.5, label="Non-Mormon")
axarr[0].set_title("KMeans Labels")
plt.legend(loc="best", shadow=True, fancybox=True)

for idx, lbl in enumerate(labels):
    if lbl == 0:#clustering.labels_[idx]:
        x1.append(Ux[idx,0])
    else:
        x2.append(Ux[idx,0])

axarr[1].hist(x2, bins=20, color='salmon', alpha=0.5, label="Non-Mormon")
axarr[1].hist(x1, bins=20, color='lightblue', alpha=0.5, label="Mormon")
axarr[1].set_title("True Labels")
axarr[1].set_xlabel("$(\max_{i}\\mathbf{v}_{i})\\times (\max_{i}\lambda_{i})$")
plt.legend(loc="center", shadow=True, fancybox=True, bbox_to_anchor = (0,0,1.65,1.75), bbox_transform=plt.gcf().transFigure)
plt.savefig("eigennorm.pdf", bbox_inches="tight")
#plt.show()


'''
plt.clf()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(Ux[:,0], Ux[:,1], c=colors)
axarr[0].set_title("KMeans Clustering")
axarr[1].scatter(Ux[:,0], Ux[:,1], c=colors2)
axarr[1].set_title("True Clustering")
axarr[1].set_xlabel("$(\max_{i}\lambda_{i})\\times (\max_{i}\mathbf{v}_{i})$")
plt.show()
'''
