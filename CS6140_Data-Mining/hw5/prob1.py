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

def PCA(A, k=10):
    U, s, V = svd(A, full_matrices=True)

    Uk = U[:,0:k]
    sk = s[:k]
    Sk = np.diag(sk)
    Vk = V[0:k,:]
    Ak = np.dot(Uk, Sk)
    Ak = np.dot(Ak, Vk)

    return Uk, Sk, Vk, Ak


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


