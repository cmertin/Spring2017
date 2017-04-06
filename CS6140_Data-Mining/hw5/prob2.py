import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def ReadFile(filename):
    matrix = []

    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        matrix.append(line.split())

    return np.asarray(matrix, dtype=np.float32)

def FrequentDirections(A, l):
    assert l%2 == 0
    
    n, m = np.size(A)

    B = np.zeros((l,m), dtype=np.float32)
    ind = np.arange(l)

    for i in range(n):
        zero_rows = ind[np.sum(np.abs(B) <= 1e-12, axis=1) == m]
        if len(zero_rows) >= 1:
            B[zero_rows[0]] = A[i]
        else:
            U, s, V = svd(B, full_matrices=False)
            delta = s[l/2 - 1] ** 2
            s = np.sqrt(np.maximum(s**2 - delta, 0))
            B = np.dot(np.diag(s), V)
    return B

mat_file = "A.dat"

A = ReadFile(mat_file)

k_list = range(1, 11)
vals = []

for k in k_list:
    Uk, Sk, Vk, Ak = PCA(A, k=k)
    temp = norm(A - Ak)
    vals.append(temp)

plt.plot(k_list, vals)
plt.xlabel("$k$")
plt.ylabel("$\\left\| A - A_{k} \\right\|_{2}$")
plt.savefig("prob1a.pdf", bbox_inches="tight")

A_10 = 0.1 * norm(A)

k_min = 1
Uk, Sk, Vk, Ak = PCA(A, k=k_min)

A_ = norm(Ak - A)

while A_ > A_10:
    k_min += 1
    Uk, Sk, Vk, Ak = PCA(A, k=k_min)
    A_ = norm(Ak - A)

print("Minimum k: ", k_min)


