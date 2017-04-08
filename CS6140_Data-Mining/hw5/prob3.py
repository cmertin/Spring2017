import numpy as np
from numpy.linalg import norm, inv
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

def NormalEquation(X, Y, lambda_=0):
    X_ = np.dot(X.T, X)
    n = np.shape(X_)[0]
    X_ = inv(X_ + lambda_**2 * np.identity(n))
    X_ = np.dot(X_, X.T)
    X_ = np.dot(X_, Y)
    return X_

def PrintTable(s, err, title=None):
    print("\\begin{table}[H]")
    print("\\centering")
    if title != None:
        print("\caption{" + str(title) + "}")
    print("\\begin{tabular}{@{}l r@{}}")
    print("\\hline\\hline")
    print("$s$ & $\\left|\\left| Y - X\cdot C\\right|\\right|_{2}$\\\\")
    print("\hline")
    for idx in range(len(s)):
        print(s[idx], "&", "%.4f" % err[idx], "\\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n")

X = ReadFile("X.dat")
Y = ReadFile("Y.dat")
lambda_ = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
err = []

for s in lambda_:
    C = NormalEquation(X, Y, lambda_=s)
    n = norm(Y - np.dot(X, C))
    err.append(n)

PrintTable(lambda_, err)

X1 = X[:66,:]
Y1 = Y[:66]
X1_ = X[66:, :]
Y1_ = Y[66:]

X2 = X[33:100,:]
Y2 = Y[33:100]
X2_ = X[:33, :]
Y2_ = Y[:33, :]

X3 = X[0:33,:]
Y3 = Y[0:33]
idx = list(range(33))
idx.extend(list(range(66, 100)))
X3 = []
Y3 = []
for i in idx:
    X3.append(X[i, :])
    Y3.append(Y[i])
X3 = np.asarray(X3)
Y3 = np.asarray(Y3)
X3_ = X[33:66, :]
Y3_ = Y[33:66]

err = []
title = "$(X_{1},\ Y_{1})$"
for s in lambda_:
    C = NormalEquation(X1, Y1, lambda_=s)
    n = norm(Y1_ - np.dot(X1_, C))
    err.append(n)

PrintTable(lambda_, err, title=title)

err = []
title = "$(X_{2},\ Y_{2})$"
for s in lambda_:
    C = NormalEquation(X2, Y2, lambda_=s)
    n = norm(Y2_ - np.dot(X2_, C))
    err.append(n)

PrintTable(lambda_, err, title=title)

err = []
title = "$(X_{3},\ Y_{3})$"
for s in lambda_:
    C = NormalEquation(X3, Y3, lambda_=s)
    n = norm(Y3_ - np.dot(X3_, C))
    err.append(n)

PrintTable(lambda_, err, title=title)




