import utils as utils
import numpy as np
import matplotlib.pyplot as plt

[t, X] = utils.loadData()
targets = utils.normalizeData(t)
X_n = utils.normalizeData(X)

TRAIN_SIZE = 100
X_n = X_n[np.arange(0, TRAIN_SIZE), :]
X_n = X_n[:, 2].reshape(100, 1)
targets = targets[np.arange(0, TRAIN_SIZE)].reshape(100, 1)

validationErrorAverage = dict()


def crossRegularization(lmd, degree):
    global validationErrorAverage
    validationError = 0
    folds = 10  # size of validation set
    for i in range(0, 10):
        xValidation = X_n[i * folds:(i + 1) * folds]
        tValidation = targets[i * folds:(i + 1) * folds]

        xTrain = np.concatenate((X_n[0: i * folds], X_n[(i+1) * folds:]))
        tTrain = np.concatenate((targets[0:i * folds], targets[(i + 1) * folds:]), 0)

        pTrain = utils.degexpand(xTrain, degree)
        w = np.dot(np.linalg.inv(np.dot(lmd, np.eye(pTrain.shape[1])) + np.dot(np.transpose(pTrain), pTrain)), np.dot(np.transpose(pTrain), tTrain))

        pValue = utils.degexpand(xValidation, degree)
        yValidation = np.dot(pValue, w)
        valError = np.mean(np.square(yValidation - tValidation))

        validationError += valError

    validationErrorAverage[lmd] = validationError / 10


lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
for lmdValue in lambdas:
    crossRegularization(lmdValue, 8)


# Produce the plot
xLabel = sorted(validationErrorAverage.keys())
error = []
for label in xLabel:
    error.append(validationErrorAverage[label])

plt.semilogx(xLabel, error)
plt.ylabel('Error')
plt.legend(['Average Validation error'])
plt.title('Polynomial degree = 8, 10-fold cross validation regularization')
plt.xlabel('lambda on log scale')
plt.show()
