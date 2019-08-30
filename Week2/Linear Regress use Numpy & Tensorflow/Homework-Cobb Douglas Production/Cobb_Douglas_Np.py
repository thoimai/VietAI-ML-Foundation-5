import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data_df = pd.read_excel('cobbdouglas.xls', header=None, skiprows=1)
# print(data_df)

data = np.asarray(data_df)
# print(data)

K = data_df[1].values
L = data_df[2].values
P = data_df[3].values
# print(P)
# print(L)
# print(K)

X = np.log(L) - np.log(K)
y = np.log(P) - np.log(K)
# print(X,y)

N = X.shape[0]
# print(N)

#Vectorized
X = X.reshape(-1,1)
X_train = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
# print(X)

# Compute Theta(weight) use Normal Equation
def compute_W(X, y):
    A = np.dot(X.T, X)
    c = np.dot(X.T, y)

    return np.dot(np.linalg.inv(A), c)


# function Gradient Descent
def np_grad_fn(W, X, y):
    N, D = X.shape
    y_hat = np.dot(X, W)
    error = y_hat - y

    return np.dot(X.T, error) / float(N)


def np_Gradient_Descent(X, y, print_every=5000,
                        niter=100000, alpha=0.01):
    '''
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    for k in range(niter):

        # TODO: Complete the below followed the above expressions
        dw = np_grad_fn(w, X, y)
        w = w - alpha * dw

        if k % print_every == 0:
            print('Weight after %d iteration: %s' % (k, str(w)))
    return w

#Run
opt_w = np_Gradient_Descent(X_train, y)