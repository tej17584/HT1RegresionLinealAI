import numpy as np
import csv
import pandas as pd
import plotly.express as px

fullData = pd.read_csv("Admission_Predict.csv")
greScore = np.array(fullData.iloc[0:300, 1].values)
toeflScore = np.array(fullData.iloc[0:300, 2].values)
SOPVar = np.array(fullData.iloc[:, 4].values)
LORVar = np.array(fullData.iloc[:, 5].values)
CGPAVar = np.array(fullData.iloc[:, 6].values)
axisY = np.array(fullData.iloc[0:300, 8].values)


TRAINING_ELEMENTS = 300
x = greScore
"""

x = np.linspace(
    -10,
    30,
    TRAINING_ELEMENTS
)

"""

X = np.vstack(
    (
        np.ones(TRAINING_ELEMENTS),
        greScore,
        #x ** 3,
    )
).T

# y = x ** 3 + 50 - 100 * np.random.rand(TRAINING_ELEMENTS)
#y = 5 + 2 * x ** 3 + np.random.randint(-15, 15, TRAINING_ELEMENTS)
dataset_1 = (X, axisY.reshape(TRAINING_ELEMENTS, 1))

# import json
# import numpy as np

# with open('dataset.json', 'r') as f:
#     dataset = json.load(f)

# X = np.array(dataset['x'])
# y = np.array(dataset['y'])
# print(X.shape, y.shape)

# dataset_1 = (X, y)
