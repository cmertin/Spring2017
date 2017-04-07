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
            sq_sv_center = s[np.floor(l/2)]**2

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

while err > A_/10:
    l += 1
    B = FrequentDirections(A, l)
    err = norm(np.dot(A.T, A) - np.dot(B.T, B))

print("Minimum l: ", l)

Ak = PCA(A, k=2)
A_ = norm(A - Ak)
err = A_
l = 0

while err > A_/10:
    l += 1
    B = FrequentDirections(A, l)
    err = norm(np.dot(A.T, A) - np.dot(B.T, B))

print("Minimum l: ", l)
