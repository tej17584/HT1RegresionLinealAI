import numpy as np


def linear_cost_derivate_regularized(X, y, theta, lambdaValue):
    m, _ = X.shape
    h = np.matmul(X, theta)
    teta_0_regularized = h
    vlambda = np.empty(300)
    vlambda.fill(lambdaValue)

    return (np.matmul((h - y).T, X).T + np.matmul(vlambda, teta_0_regularized)) / m
