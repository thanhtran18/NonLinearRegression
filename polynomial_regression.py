from utils import loadData, normalizeData, degexpand
import numpy as np
import matplotlib.pyplot as plt


# LOAD THE DATA AND EDIT IT:
# ==========================
[t, X] = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

# CREATE THE TRAIN AND TEST SETS:
# ================================
TRAIN_SIZE = 100  # number of training examples
xTrain = X_n[np.arange(0, TRAIN_SIZE), :]  # training input data
tTrain = t[np.arange(0, TRAIN_SIZE)]  # trainint output data
xTest = X_n[np.arange(TRAIN_SIZE, X_n.shape[0]), :]  # testing input data
tTest = t[np.arange(TRAIN_SIZE, X_n.shape[0])]  # testing output data

trainErrors = [0] * 10
testErrors = [0] * 10
for i in np.arange(1, 11):

    pTrain = degexpand(xTrain, i)
    w = np.dot(np.linalg.pinv(np.dot(np.transpose(pTrain), pTrain)), np.dot(np.transpose(pTrain), tTrain))

    trainDifference = np.dot(pTrain, w) - tTrain
    trainError = (np.mean(np.square(trainDifference)))

    pTest = degexpand(xTest, i)
    testDifference = np.dot(pTest, w) - tTest
    testError = (np.mean(np.square(testDifference)))

    trainErrors[i-1] = trainError
    testErrors[i-1] = testError

# Produce a plot of results
plt.plot(range(1, 11), trainErrors, "r-")
plt.plot(range(1, 11), testErrors, "b-")
plt.ylabel('Error')
plt.legend(['Training error', 'Test error'])
plt.title('Fig degree %d polynomial' % 5)
plt.xlabel('Polynomial degree')
plt.show()



