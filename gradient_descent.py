import numpy as np


def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        alpha=0.00001,
        treshold=0.001,
        max_iter=10000,
        lambdaValue=0.5):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta, lambdaValue)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta, lambdaValue)
        i += 1
        costs.append(cost(X, y, theta, lambdaValue))
        gradient_norms.append(cost_derivate(X, y, theta, lambdaValue))
    return theta, costs, gradient_norms
