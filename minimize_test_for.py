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
theta_0 = np.random.rand(n, 1)
theta_0_burned = [[0.046054599], [0.082545956]]
print(theta_0)

def frange2(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0
    else:
        start += 0.0  # force it to be a float

    if inc == None:
        inc = 1.0

    count = int((end - start) / inc)
    if start + count * inc != end:
        # need to adjust the count.
        # AFAIKT, it always comes up one short.
        count += 1

    L = [None, ] * count
    for i in range(count):
        L[i] = start + i * inc

    return L


for i in frange2(0, 30, 0.5):

    lambdaValue = i
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

    plt.plot(X[:, 1], np.matmul(X, theta), color='red',
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
