from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.cluster import KMeans
from math import sqrt
from random import choice, shuffle
from pyclustering.cluster.kmedians import kmedians


def ReadFile(filename):
    data = []
    clusters = []
    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        temp = np.asarray(line.split()[1:], dtype=float)
        data.append(temp)
        clusters.append(int(line.split()[0]))

    return np.asarray(data), clusters

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

def MeanCost_(data, centroids, segments):
    cost_list = []
    N = len(data)
    for idx, c in enumerate(centroids):
        cost = 0
        for i, s in enumerate(segments[idx]):
            cost += norm(c - data[s])
        cost = cost/N
        cost_list.append(cost)
    return cost_list

filename = "C3.txt"
n_clusters = 4
data, clusters = ReadFile(filename)
itr = 1000
best_cost = []
all_cost = []
clusters = [[] for x in range(n_clusters)]
b_centroids = []

flag = False

b_cost = np.inf

for i in range(itr):
    d = data.copy()
    centroids = []
    shuffle(d)

    for j in range(len(clusters)):
        clusters[j] = choice(d)
        shuffle(d)
    k_medians = kmedians(d, clusters)
    k_medians.process()
    segments = k_medians.get_clusters()
    for j in range(len(segments)):
        temp = int(np.median(segments[j]))
        centroids.append(data[temp][:])
    cost = MeanCost_(data, centroids, segments)
    if len(cost) == n_clusters:
        t_cost = norm(cost)
        if t_cost < b_cost:
            b_cost = t_cost
            best_cost = cost
            b_centroids = centroids
        all_cost.append(cost)

print('\n')
print(b_centroids)
print(b_cost)
print(best_cost)

costs = [[] for x in range(n_clusters)]

for cost in all_cost:
    for idx, c in enumerate(cost):
        costs[idx].append(c)

for idx, c in enumerate(costs):
    c = np.sort(c)
    c2 = c/(np.asarray(c)*0.1).sum()
    c2 = np.cumsum(c2 * 0.1)
    legend = "Cluster " + str(idx + 1)
    plt.plot(c, c2, label=legend)

plt.xlabel("Mean Cost")
plt.ylabel("%")
plt.title("K-Median Mean Cost for n_clusters = " + str(n_clusters))
plt.ylim([0,1.05])
fileout = "cdf_median_n-" + str(n_clusters) + ".pdf"
plt.legend(loc="best")
plt.savefig(fileout, bbox_inches="tight")
plt.show()

'''
0.605055333826
[0.30330758172655836, 0.23772694566275099, 0.41139420130511462, 0.21985717704224819]

0.58697025661
[0.20568306847780832, 0.24326922630466974, 0.12857561094812447, 0.32800964580311959, 0.3448573988599406]
'''
