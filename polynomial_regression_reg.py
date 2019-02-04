import utils as utils
import numpy as np
import matplotlib.pyplot as plt

[t, X] = utils.loadData()
targets = utils.normalizeData(t)
X_n = utils.normalizeData(X)

TRAIN_SIZE = 100
X_n = X_n[np.arange(0, TRAIN_SIZE), :]
targets = targets[np.arange(0, TRAIN_SIZE)]

validationErrorAverage = dict()


def crossRegularization(lmd, degree):

    validationError = 0
    folds = 10  # size of validation set
    for i in range(0, 10):
        xValidation = X_n[i * folds:(i + 1) * folds, :]
        tValidation = targets[i * folds:(i + 1) * folds]

        xTrain = np.concatenate((X_n[0:i * folds, :], X_n[(i + 1) * folds:, :]), 0)
        tTrain = np.concatenate((targets[0:i * folds], targets[(i + 1) * folds:]), 0)

        pTrain = utils.degexpand(xTrain, degree)
        w = np.linalg.inv(lmd * np.identity(pTrain.shape[1]) + np.transpose(pTrain).dot(pTrain)).dot(np.transpose(pTrain)).dot(tTrain)

        pValue = utils.degexpand(xValidation, degree)
        yValidation = np.dot(np.transpose(w), np.transpose(pValue))
        tValidationDifference = tValidation - np.transpose(yValidation)
        validationError = np.sqrt(np.mean(np.square(tValidationDifference)))

        validationError += validationError

    validationErrorAverage[lmd] = validationError / 10


lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
for lmdValue in lambdas:
    crossRegularization(lmdValue, 8)


# Produce a plot of results.
label = sorted(validationErrorAverage.keys())
error = []
for key in label:
    error.append(validationErrorAverage[key])

plt.semilogx(label, error)
plt.ylabel('Error')
plt.legend(['Average Validation error'])
plt.title('Polynomial degree = 8, 10-fold cross validation regularization')
plt.xlabel('lambda on log scale')
plt.show()
