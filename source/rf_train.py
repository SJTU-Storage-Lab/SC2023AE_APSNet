from sklearn.ensemble import RandomForestClassifier as RFC
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

from utils import model_and_dataset_selection

n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict = model_and_dataset_selection.train_select_offline()

n = {5: 10000, 7: 10000, 15: 10000, 30: 10000, 45: 10000, 60: 10000, 90: 10000, 120: 10000}


def loadData():

    if data_type != 'D':
        X_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', allow_pickle=True)
        y_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', allow_pickle=True)
    else:
        X_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train_rf.npy', allow_pickle=True)
        y_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels_rf.npy', allow_pickle=True)
    X_test = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', allow_pickle=True)
    y_test = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', allow_pickle=True)
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
    print(metrics.classification_report(true, predicted, digits=5))


X_train, y_train, X_test, y_test = loadData()
print(X_train.shape, X_test.shape)

print('------------------ Loading Data ------------------')
X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

print('------------------ Random Forest ------------------')
if (data_folder_name_dict[data_type] == 'mc1_mc2'):
    if (n_days_lookahead == 5):
        model_rf = RFC(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    elif (n_days_lookahead == 7):
        model_rf = RFC(criterion='entropy', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=150)
    elif (n_days_lookahead == 15):
        model_rf = RFC(criterion='entropy', max_depth=100, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=150)
    elif (n_days_lookahead == 30):
        model_rf = RFC(criterion='entropy', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    elif (n_days_lookahead == 45):
        model_rf = RFC(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    elif (n_days_lookahead == 60):
        model_rf = RFC(criterion='entropy', max_depth=100, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=3, n_estimators=150)
    elif (n_days_lookahead == 90):
        model_rf = RFC(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=3, n_estimators=200)
    elif (n_days_lookahead == 120):
        model_rf = RFC(criterion='gini', max_depth=100, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
else:
    model_rf = RFC()
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
print_all_metrics(y_test, y_pred)
joblib.dump(model_rf, '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/rf.pkl')
print('Done')
