from __future__ import print_function, division
import numpy as np
from copy import deepcopy
from numpy.random import permutation
from numpy.linalg import eig, matrix_power
import random
import matplotlib.pyplot as plt

def ReadMatrix(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    matrix = []
    for line in lines:
        matrix.append(np.asarray(line.split(), dtype=np.float64))

    return np.asarray(matrix)

def Mt(M, t):
    return matrix_power(M,t)

def MatrixPower(M, t, q0):
    M2 = deepcopy(M)
    M_t = Mt(M2, t)
    q_star = np.dot(M_t, q0)
    return q_star

def StatePropogation(M, t, q0):
    M2 = deepcopy(M)
    q2 = deepcopy(q0)

    for i in range(t):
        q2 = np.dot(M,q2)

    return q2

def NextNode(col):
    num_range = random.uniform(0, 1)
    rows = np.shape(col)[0]
    search_order = permutation(list(range(rows)))
    idx = None
    for i in range(rows):
        idx_ = search_order[i]
        col_search = col[idx_]
        gr8_zero = col_search > 0
        le_range = col_search < num_range
        if gr8_zero and le_range:
            idx = idx_
            break
        while idx == None:
            idx = NextNode(col)
    return idx

def RandomWalk(M, t, t0, q0):
    # Burn in period
    q_in = deepcopy(q0)
    q_0 = deepcopy(q0)
    M2 = deepcopy(M)
    M_t = Mt(M2, t0)
    q_i = np.dot(M_t,q_0)
    idx = NextNode(M[:,0])

    for i in range(t0):
        q_0 = np.zeros((np.shape(M)[1],1))
        q_0[idx] = 1
        idx = NextNode(M[:,idx])

    state_vec = np.zeros((np.shape(M)[1],1))
    for i in range(len(q_0)):
        if int(q_0[i]) > 0:
            idx = i

    M_t = Mt(deepcopy(M),t)
    for i in range(t):
        qi = np.dot(M_t, q_0)
        idx = NextNode(M[:,idx])
        state_vec[idx] += 1
    return 1.0/t * state_vec
        

def EigenAnalysis(M):
    M2 = deepcopy(M)
    w, v = eig(M)
    v = np.asarray([v[:,0]])
    return v.T

M_file = "M.dat"
L_file = "L.dat"
t0 = 100
t = 1024

M = ReadMatrix(M_file)
L = ReadMatrix(L_file)

n,m = np.shape(M)
q0 = np.zeros((m,1), dtype=float)
q0[0] = 1.0

# Problem A
matrix_power_q = MatrixPower(M, t, q0)
state_propogation_q = StatePropogation(M, t, q0)
random_walk_q = RandomWalk(M, t, t0, q0)
eigen_analysis_q = EigenAnalysis(M)

if False:
    print("Matrix Power")
    print("============")
    print(matrix_power_q)

    print("\nState Propogation")
    print("=================")
    print(state_propogation_q)

    print("\nRandom Walk")
    print("===========")
    print(random_walk_q)

    print("\nEigenvalues")
    print("===========")
    print(eigen_analysis_q)

# Problem B
q0 = np.ones(np.shape(M)[1]) * 0.1
m_pow_norm = []
state_prop_norm = []
t_vals = []

for i in range(1, 200):
    t_vals.append(i)
    matrix_power_qb = MatrixPower(deepcopy(M), i, deepcopy(q0))
    state_propogation_qb = StatePropogation(deepcopy(M), i, deepcopy(q0))
    diff = np.linalg.norm(matrix_power_qb - matrix_power_q[:,0])/np.linalg.norm(matrix_power_q[:,0])
    m_pow_norm.append(diff)
    diff = np.linalg.norm(state_propogation_qb - state_propogation_q[:,0])/np.linalg.norm(state_propogation_q[:,0])
    state_prop_norm.append(diff)

plt.plot(t_vals, m_pow_norm, label="Matrix Power", linestyle="dotted")
plt.plot(t_vals, state_prop_norm, label="State Propogation", linestyle="dashed")
plt.xlabel("t")
plt.ylabel("Normed Difference")
#plt.ylim([-0.5,2])
plt.legend(loc="upper right")
plt.title("Normed Difference Between True and Computed")
plt.savefig("partb.pdf", bbox_inches="tight")

if False:
    print("Matrix Power")
    print("============")
    print(matrix_power_qb)

    print("\nState Propogation")
    print("=================")
    print(state_propogation_qb)
