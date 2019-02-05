import utils as utils
import numpy as np
import matplotlib.pyplot as plt

[t, X] = utils.loadData()
X_n = utils.normalizeData(X)
t = utils.normalizeData(t)

# CREATE THE TRAIN AND TEST SETS:
# ================================
TRAIN_SIZE = 100  # number of training examples
xTrain = X_n[np.arange(0, TRAIN_SIZE), :]  # training input data
tTrain = t[np.arange(0, TRAIN_SIZE)]  # training output data
xTest = X_n[np.arange(TRAIN_SIZE, X_n.shape[0]), :]  # testing input data
tTest = t[np.arange(TRAIN_SIZE, X_n.shape[0])]  # testing output data
tTest = tTest.reshape(292, 1)


trainErrors = dict()
testErrors = dict()

xTrainFeature = xTrain[:, 2].reshape(100, 1)
xTestFeature = xTest[:, 2].reshape(292, 1)
degrees = [2, 10, 12]

# calculate train and test error for each feature with different polynomial degrees
for i in degrees:

    pTrain = utils.degexpand(xTrainFeature, i)
    w = np.dot(np.linalg.pinv(pTrain), tTrain)

    trainDifference = tTrain - np.transpose(np.dot(np.transpose(w), np.transpose(pTrain)))
    trainError = np.sqrt(np.mean(np.square(trainDifference)))

    pTest = utils.degexpand(xTestFeature, i)
    testDifference = tTest - np.transpose(np.dot(np.transpose(w), np.transpose(pTest)))
    testError = np.mean(np.square(testDifference))

    trainErrors[i] = trainError
    testErrors[i] = testError

    # Produce the plots
    x_ev = np.arange(min(xTrainFeature), max(xTrainFeature) + 0.1, 0.1)
    pEv = utils.degexpand(x_ev.reshape(x_ev.shape[0], 1), i)
    y_ev = np.dot(pEv, w)
    rows = (xTrainFeature.max() + 0.1 - xTrainFeature.min())/0.1

    plt.plot(x_ev, y_ev, 'r.-')
    plt.plot(xTrainFeature, tTrain, 'gx', markersize=10)
    plt.plot(xTestFeature, tTest, 'bo', markersize=10, mfc='none')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Fig degree %d polynomial' % i)
    plt.show()
