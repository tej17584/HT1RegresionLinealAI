import numpy as np
from matplotlib import pyplot as plt

from datasets import dataset_1
from gradient_descent import gradient_descent
from linear_cost import linear_cost
from linear_cost_derivate import linear_cost_derivate

from ln_regularizado import linear_cost_regularized
from linear_cost_derivate_regularized import linear_cost_derivate_regularized

# Training data
(X, y) = dataset_1
m, n = X.shape
theta_burned = [[-3.77130144], [0.01401026]]
theta_0 = np.random.rand(n, 1)
theta_0_burned = [[0.6503114], [0.14125331]]


lambdaValue = 25.0

theta, costs, gradient_norms = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost_regularized,
    linear_cost_derivate_regularized,
    alpha=0.000001,
    treshold=0.000001,
    max_iter=10000,
    lambdaValue=lambdaValue,
)

# Plot training data
plt.scatter(X[:, 1], y)

plt.plot(X[:, 1], np.matmul(X, theta_burned), color='red',
         label="Thetha :" + str(theta) + "  lambdaValue: "+str(lambdaValue))

# plt.plot(np.arange(len(costs)), costs)
plt.legend()

plt.show()

# # X => (11, 2)
# Xtraining = X[0:5, :]
# ytraining = y[0:5, :]
# Xcv = X[5:8, :]
# ycv = y[5:8, :]
# Xtest = X[8:, :]
# ytest = y[8:, :]

# # Model deduction
# theta = np.linalg.lstsq(Xtraining, ytraining)[0]

# print("Jtraining:", linear_cost(theta, Xtraining, ytraining))
# print("Jcv:", linear_cost(theta, Xcv, ycv))
