from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy
import scipy.cluster.hierarchy as sch


def ReadFile(filename):
    data = []
    clusters = []
    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        temp = np.asarray(line.split()[1:], dtype=float)
        data.append(temp)
        clusters.append(int(line.split()[0]))

    return data, clusters

filename = "C1.txt"
alg = "Mean-Link"
alg_type1 = "average" #["average", "complete", "single"]
alg_type2 = "average" #["average", "complete", "ward"]
out_plot = alg + ".pdf"
data, clusters = ReadFile(filename)
n_clusters = 4

c = AgglomerativeClustering(n_clusters=n_clusters, linkage=alg_type2).fit(data)
labels = c.labels_

d = sch.distance.pdist(data)
Z = sch.linkage(d, method=alg_type1)
plt.figure()
dn = sch.dendrogram(Z, labels=labels)
sch.set_link_color_palette(['m','c','y','k'])
dn1 = sch.dendrogram(Z, labels=labels)
plt.title("Hierarchial Dendrogram for " + alg)
plt.savefig(alg + "_dendro.pdf", bbox_inches="tight")
plt.show()

for l in np.unique(labels):
    x = []
    y = []
    for l_ in range(len(labels)):
        if l == labels[l_]:
            x.append(data[l_][0])
            y.append(data[l_][1])
    plt.scatter(x, y, color=plt.cm.jet(np.float(l) / np.max(labels + 1)))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Hierarchial Clustering: " + alg)
plt.savefig(out_plot, bbox_inches="tight")
#plt.show()

