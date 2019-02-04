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
tTrain = t[np.arange(0, TRAIN_SIZE)]  # trainint output data
xTest = X_n[np.arange(TRAIN_SIZE, X_n.shape[0]), :]  # testing input data
tTest = t[np.arange(TRAIN_SIZE, X_n.shape[0])]  # testing output data
tTest = tTest.reshape(292, 1)


trainErrors = dict()
testErrors = dict()

xTrainFeature = xTrain[:, 2].reshape(100, 1)
xTestFeature = xTest[:, 2].reshape(292, 1)
degrees = [1, 2, 10, 12]


def predict(weights, x, deg):
    return [weights[i] * x[i] ** deg for i in range(0, w.shape[0] + 1)]
    # return np.sqrt(np.mean(np.square(testDifference)))

# calculate train and test error for each feature with polynominal degree = 0
for i in degrees:
# index = 2
#     xTrainFeature = xTrain[:, i].reshape(100, 1)
#     xTestFeature = xTest[:, i].reshape(292, 1)

    pTrain = utils.degexpand(xTrainFeature, i)
    w = np.dot(np.linalg.pinv(pTrain), tTrain)

    # yTrain = np.dot(np.transpose(w), np.transpose(pTrain))
    trainDifference = tTrain - np.transpose(np.dot(np.transpose(w), np.transpose(pTrain)))
    trainError = np.sqrt(np.mean(np.square(trainDifference)))

    pTest = utils.degexpand(xTestFeature, i)
    # yTest = np.dot(np.transpose(w), np.transpose(pTest))
    testDifference = tTest - np.transpose(np.dot(np.transpose(w), np.transpose(pTest)))
    testError = np.sqrt(np.mean(np.square(testDifference)))

    trainErrors[i] = trainError
    testErrors[i] = testError

    x_ev = np.arange(min(xTrainFeature), max(xTrainFeature) + 0.1, 0.1)
    pEv = utils.degexpand(x_ev.reshape(x_ev.shape[0], 1), i)
    y_ev = np.dot(pEv, w)
    rows = (xTrainFeature.max() + 0.1 - xTrainFeature.min())/0.1
    # y_ev = np.arange(np.dot(x), w.max() + 0.1, (w.max() + 0.1 - w.min())/rows)  # put your regression estimate here
    # y_ev = np.dot(x_ev.reshape(x_ev.shape[0], 1), np.transpose(w.reshape(w.shape[0], 1)))


    plt.plot(x_ev, y_ev, 'r.-')
    # plt.plot(np.transpose(x_ev.reshape(x_ev.shape[0], 1)), np.transpose(y_ev.reshape(y_ev.shape[0], 1)), 'r.-')
    # plt.plot(np.transpose(x_ev.reshape(x_ev.shape[0], 1)), np.transpose(testError.reshape(testError.shape[0], 1)), 'r.-')
    plt.plot(xTrainFeature, tTrain, 'gx', markersize=10)
    plt.plot(xTestFeature, tTest, 'bo', markersize=10, mfc='none')


    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Fig degree %d polynomial' % i)


    plt.show()
