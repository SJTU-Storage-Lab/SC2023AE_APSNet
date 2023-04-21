from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn import metrics

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

n = {5: 20000, 7: 20000, 15: 20000, 30: 20000, 45: 20000, 60: 20000, 90: 20000, 120: 20000}

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Unbalanced}:  '))

if(data_type not in ['A', 'B', 'C', 'D']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'unbalanced'}


def loadData():

    if dit_str[data_type] not in ('unbalanced'):
        X_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', allow_pickle=True)
        y_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', allow_pickle=True)
    else:
        X_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train_rf.npy', allow_pickle=True)
        y_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels_rf.npy', allow_pickle=True)
    X_test = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', allow_pickle=True)
    y_test = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', allow_pickle=True)
    X_train = X_train[0:n[n_days_lookahead]].astype('float32')
    y_train = y_train[0:n[n_days_lookahead]].astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    return X_train, y_train, X_test, y_test


def print_all_metrics(true, predicted):
    print(metrics.classification_report(true, predicted, digits=5), metrics.roc_auc_score(true, predicted))


X_train, y_train, X_test, y_test = loadData()

print('------------------ Loading Data ------------------')
X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

print('------------------ Support Vector ------------------')
model_svc = SVC(probability=True, C=1000, gamma=0.5, kernel='rbf')
model_svc.fit(X_train, y_train)
y_pred = model_svc.predict(X_test)
# print_all_metrics(y_test,y_pred)
joblib.dump(model_svc, '../trained_model/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/svm.pkl')
print('Done')
