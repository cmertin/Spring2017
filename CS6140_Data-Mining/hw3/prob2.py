from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.cluster import KMeans
from math import sqrt

def ReadFile(filename):
    data = []
    clusters = []
    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        temp = np.asarray(line.split()[1:], dtype=float)
        data.append(temp)
        clusters.append(int(line.split()[0]))

    return data, clusters

def Gonzalez(X, k=3):
    phi = np.zeros(len(X), dtype=int)
    c = [[] for x in range(k)]
    c[0] = X[0]

    for i in range(1, k):
        M = 0
        c[i] = X[0]
        for j in range(0, len(X)):
          if norm(X[j] - c[phi[j]]) > M:
              M = norm(X[j] - c[phi[j]])
              c[i] = X[j]
        for j in range(0, len(X)):
            if norm(X[j] - c[phi[j]]) > norm(X[j] - c[i]):
                phi[j] = i
    return phi, c

def CenterCost(data, centers):
    cost_list = []
    for c in centers:
        cost = 0
        for d in data:
            c_ = norm(d - c)
            if c_ > cost:
                cost = c_
        cost_list.append(cost)
    return cost_list

def MeanCost(data, centers):
    cost_list = []
    N = len(data)
    for c in centers:
        cost = 0
        for d in data:
            cost += norm(d - c)**2
        cost_list.append(cost/N)
    return cost_list

filename = "C2.txt"
n_clusters = 3
data, clusters = ReadFile(filename)
phi, c = Gonzalez(data, k=n_clusters)
center_gonzalez = np.asarray(c.copy())
phi_ = np.unique(phi)

for l in phi_:
    x = []
    y = []
    for l_ in range(len(phi)):
        if phi[l_] == l:
            x.append(data[l_][0])
            y.append(data[l_][1])
    plt.scatter(x, y, color=plt.cm.jet(np.float(l) / np.max(phi + 1)))

for c_ in c:
    x = c_[0]
    y = c_[1]
    plt.scatter(x, y, color="r", alpha=0.60)

plt.xlabel("x")
plt.ylabel("y")
plt.title("K Means: Gonzalez")
plt.savefig("gonzalez.pdf", bbox_inches="tight")
#plt.show()

print(c)

cost = CenterCost(data, c)
print(cost)

cost = MeanCost(data, c)
print(cost)

costs2 = [[] for x in range(n_clusters)]


costs = [[] for x in range(n_clusters)]

for i in range(200):
    k_means = KMeans(n_clusters = n_clusters, init="random", algorithm="full", n_init=1, copy_x=True).fit(data)
    cost = MeanCost(data, k_means.cluster_centers_)
    k_means2 = KMeans(n_clusters = n_clusters, init=k_means.cluster_centers_, copy_x=True, algorithm="full").fit(data)
    cost2 = MeanCost(data, k_means2.cluster_centers_)
    for j in range(len(cost)):
        costs[j].append(cost[j])
        costs2[j].append(cost2[j])

        
plt.clf()        
for idx, cost in enumerate(costs):
    c1 = np.sort(cost)
    c1_ = c1/(np.asarray(c1)*0.1).sum()
    c1_ = np.cumsum(c1_ * 0.1)
    legend = "Cluster " + str(idx + 1)
    plt.plot(c1, c1_, label=legend)

plt.legend(loc="best")
plt.ylim([0,1.05])
plt.xlabel("Mean Cost")
plt.ylabel("%")
plt.title("K-Means++ Mean Cost Cumulative Density Plot")
kmeans_centers = k_means.cluster_centers_
#plt.savefig("cdf.pdf", bbox_inches="tight")
#plt.show()
    
k_means = KMeans(n_clusters=n_clusters, init="random", algorithm="full", copy_x=True).fit(data)

labels = k_means.labels_
labels_ = np.unique(labels)

plt.clf()
for l in labels_:
    x = []
    y = []
    for l_ in range(len(labels)):
        if l == labels[l_]:
            x.append(data[l_][0])
            y.append(data[l_][1])
    plt.scatter(x, y, color=plt.cm.jet(np.float(l) / np.max(labels_ + 1)))

for c in k_means.cluster_centers_:
    x = c[0]
    y = c[1]
    plt.scatter(x, y, color="r", alpha=0.75)

plt.xlabel("x")
plt.ylabel("y")
plt.title("K Means: K-Means++")
plt.savefig("kmeanspp.pdf", bbox_inches="tight")
#plt.show()

c_123 = np.asarray([data[0], data[1], data[2]])

k_means = KMeans(n_clusters=n_clusters, algorithm="full", copy_x=True, init=c_123).fit(data)
cost = MeanCost(data, k_means.cluster_centers_)
print(cost)
print(k_means.cluster_centers_)

k_means = KMeans(n_clusters=n_clusters, algorithm="full", copy_x=True, init=center_gonzalez).fit(data)
cost = MeanCost(data, k_means.cluster_centers_)
print(cost)
print(k_means.cluster_centers_)

plt.clf()
for idx, cost in enumerate(costs2):
    c1 = np.sort(cost)
    c1_ = c1/(np.asarray(c1)*0.1).sum()
    c1_ = np.cumsum(c1_ * 0.1)
    legend = "Cluster " + str(idx + 1)
    plt.plot(c1, c1_, label=legend)

plt.legend(loc="best")
plt.ylim([0,1.05])
plt.xlabel("Mean Cost")
plt.ylabel("%")
plt.title("K-Means Lloyd's Algorithm with K-Means++ Initializer")
plt.savefig("kmeanspp_lloyds.pdf", bbox_inches="tight")
#plt.show()
