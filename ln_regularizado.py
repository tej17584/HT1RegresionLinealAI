import numpy as np


def linear_cost_regularized(X, y, theta, lambdaValue):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    teta_0_regularized = h ** 2

    return (sq.sum() + (lambdaValue * teta_0_regularized.sum())) / (2 * m)
