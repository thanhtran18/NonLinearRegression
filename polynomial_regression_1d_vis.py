# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:17:49 2016

@author: linhb
"""

# !/usr/bin/env python

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
degrees = [2, 9, 10]
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

    x_ev = np.arange(xTrainFeature.min(), xTrainFeature.max() + 0.1, 0.1)
    rows = (xTrainFeature.max() + 0.1 - xTrainFeature.min())/0.1
    y_ev = np.arange(w.min(), w.max() + 0.1, (w.max() + 0.1 - w.min())/rows)  # put your regression estimate here

    plt.plot(x_ev, y_ev, 'r.-')
    plt.plot(xTrainFeature, tTrain, 'gx', markersize=10)
    plt.plot(xTestFeature, tTest, 'bo', markersize=10, mfc='none')


    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Fig degree %d polynomial' % i)

    # Produce a plot of results.
    # plt.bar(np.arange(X_n.shape[1]), [float(v) for v in trainErrors.values()], 0.33,
    #         color='blue',
    #         label='Train Error')
    # plt.bar(np.arange(X_n.shape[1]) + 0.33, [float(v) for v in testErrors.values()], 0.33,
    #         color='green',
    #         label='Test Error')
    #
    # plt.plot([float(k) for k in trainErrors.keys()], [float(v) for v in trainErrors.values()])
    # plt.plot([float(k) for k in testErrors.keys()], [float(v) for v in testErrors.values()])
    # # plt.xticks(np.arange(X_n.shape[1]) + 0.33, [('F' + str(k)) for k in trainErrors.keys()])
    # plt.ylabel('Error')
    # plt.legend(['Training error', 'Test error'])
    # plt.title('Using Single Feature')
    # plt.xlabel('Feature (F)')
    plt.show()











"""
x_train = X_n[0:TRAIN_SIZE, 2].reshape(100, 1)
x_test = X_n[TRAIN_SIZE:, 2].reshape(292, 1)

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace((min(x_train)).item(), (max(x_train)).item(), num=500)

# TO DO:: Put your regression estimate here in place of x_ev.
theta_train = utils.degexpand(x_train, 3)
w = np.dot(np.linalg.pinv(theta_train), tTrain)
theta_ev = utils.degexpand(np.transpose(np.asmatrix(x_ev)), 3)
y_ev = np.dot(np.transpose(w), np.transpose(theta_ev))

# Evaluate regression on the linspace samples.
# y_ev = np.random.random_sample(x_ev.shape)
# y_ev = 100*np.sin(x_ev)


plt.plot(x_train, tTrain, 'bo')
plt.plot(x_test, tTest, 'go')
plt.plot(x_ev, np.transpose(y_ev), 'r.-')
plt.legend(['Training data', 'Test data', 'Learned Polynomial'])
plt.title('A visualization of a regression estimate using random outputs')
plt.show()
"""