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
tTest = tTest.reshape(292, 1)

trainErrors = dict()
testErrors = dict()
for i in np.arange(1, 11):

    pTrain = degexpand(xTrain, i)
    w = np.dot(np.linalg.pinv(pTrain), tTrain)

    trainDifference = tTrain - np.transpose(np.dot(np.transpose(w), np.transpose(pTrain)))
    trainError = np.sqrt(np.mean(np.square(trainDifference)))

    pTest = degexpand(xTest, i)
    testDifference = tTest - np.transpose(np.dot(np.transpose(w), np.transpose(pTest)))
    testError = np.sqrt(np.mean(np.square(testDifference)))

    trainErrors[i] = trainError
    testErrors[i] = testError

# Produce a plot of results.
plt.plot([float(k) for k in trainErrors.keys()], [float(v) for v in trainErrors.values()])
plt.plot([float(k) for k in testErrors.keys()], [float(v) for v in testErrors.values()])
plt.ylabel('Error')
plt.legend(['Training error', 'Test error'])
plt.title('Fig degree %d polynomial' % 5)
plt.xlabel('Polynomial degree')
plt.show()



