import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.random import normal
from sklearn import random_projection
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

    return Ak#Uk, Sk, Vk, Ak

def FrequentDirections(A, l):
    n, m = np.shape(A)

    if np.floor(l/2) >= m:
        raise ValueError("Error: \'l\' must be smaller than m*2")
    if l >= n:
        raise ValueError("Error: \'l\' must not be greater than n")

    # Initialize output matrix B
    B = np.zeros([l,m])

    # Compute zero valued row list
    zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(B, axis = 1)])[0].tolist()

    # Repeat inserting each row of matrix A
    for i in range(n):
        # Insert a row into matrix B
        B[zero_rows[0], :] = A[i, :]

        # Remove zero valued row from the list
        zero_rows.remove(zero_rows[0])

        # If there is no more zero valued rows
        if len(zero_rows) == 0:

            # Compute SVD of matrix B
            U, s, V = svd(B, full_matrices=False)

            # Obtain squared singular value for threshold
            sq_sv_center = s[int(np.floor(l/2))]**2

            # Update sigma to shrink the row norms
            sigma_tilda = [(0.0 if d < 0.0 else np.sqrt(d)) for d in (s**2 - sq_sv_center)]

            # Update matrix B where at least half the rows are all zero
            B = np.dot(np.diagflat(sigma_tilda), V)

            # Update the zero valued row list
            zero_rows = np.nonzero([round(s, 7) == 0 for s in np.sum(B, axis=1)])[0].tolist()
    return B

mat_file = "A.dat"

A = ReadFile(mat_file)

A_ = norm(A)**2
err = A_
l = 0
l_list = []
err_list = []

while err > A_/10:
    l += 1
    B = FrequentDirections(A, l)
    err = norm(np.dot(A.T, A) - np.dot(B.T, B))
    l_list.append(l)
    err_list.append(err)

A_10_lst = [A_/10]
A_10_lst = A_10_lst * len(l_list)

plt.plot(l_list, err_list, label="$\\left|\\left| A^{T}A - B^{T}B\\right|\\right|_{2}$")
plt.plot(l_list, A_10_lst, label="$\\left|\\left| A\\right|\\right|_{2}^{2}/10$")
plt.xlabel("$\ell$")
plt.ylabel("err")
plt.legend(loc="best")
plt.savefig("prob2a.pdf", bbox_inches="tight")

print("Minimum l: ", l)

Ak = PCA(A, k=2)
A_ = norm(A - Ak)**2
err = A_
l = 0

while err > A_/10:
    l += 1
    B = FrequentDirections(A, l)
    err = norm(np.dot(A.T, A) - np.dot(B.T, B))

print("Minimum l: ", l)

def RandomVals(l,d):
    return 1.0/np.sqrt(l) * np.asarray([normal() for i in range(d)])

l_vals = []
for i in range(1000):
    l = 0

    A_ = norm(A)**2/10
    err = norm(A)*norm(A)
    while err > A_:
        l += 1
        S = [RandomVals(l, np.shape(A)[0]) for i in range(l)]    
        S = np.asarray(S)
        B = np.dot(S, A)
        err = norm(np.dot(A.T, A) - np.dot(B.T, B))
    l_vals.append(int(l))

l_vals = np.asarray(np.sort(l_vals))
l_ = l_vals/(0.1 * l_vals.sum())
l_ = np.cumsum(l_ * 0.1)

plt.clf()
plt.plot(l_vals, l_)
plt.xlabel("value")
plt.ylabel("percentage")
plt.savefig("prob2b.pdf", bbox_inches="tight")
