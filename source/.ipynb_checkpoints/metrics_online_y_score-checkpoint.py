import numpy as np
from sklearn import metrics as sklearn_metrics
import joblib
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.neighbors import KNNClassifier
from river.linear_model import LogisticRegression
from river import optim
from river.ensemble import AdaptiveRandomForestClassifier
from river.linear_model import PAClassifier

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2}: '))

if(data_type not in ['A', 'B', 'C']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'mc1_no_aging_attr', 'E': 'balanced_mc1_mc2', 'F': '500_mc2'}

model_type = str(input('Please input the type of trained model to use {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2}: '))

if(model_type not in ['A', 'B', 'C']):
    print('Input does not meet requirements.')
    exit()


def loadData():

    X = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', allow_pickle=True)
    y = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', allow_pickle=True)

    X = X.astype('float32')
    y = y.astype('float32')
    return X.reshape((len(X), 30, -1)), y


rf = joblib.load('../trained_model/' + dit_str[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/rf_online.pkl')
dt = joblib.load('../trained_model/' + dit_str[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/dt_online.pkl')
lr = joblib.load('../trained_model/' + dit_str[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/lr_online.pkl')
pac = joblib.load('../trained_model/' + dit_str[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pac_online.pkl')
knn = joblib.load('../trained_model/' + dit_str[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/knn_online.pkl')

X_test, y_test = loadData()
X_test = X_test.reshape((len(X_test), -1))
print(X_test.shape)

headers = [str(i) for i in range(330)]
data_x_test = [dict(zip(headers, x)) for x in X_test]
data_y_test = [True if y == 1 else False for y in y_test]
y_true = []
y_pred_rf, y_pred_dt, y_pred_lr, y_pred_pac, y_pred_knn = [], [], [], [], []
y_score_rf, y_score_dt, y_score_lr, y_score_pac, y_score_knn = [], [], [], [], []
i = 0
for Xi, yi in zip(data_x_test, data_y_test):
    if i % 1000 == 0:
        print(i)
    i += 1

    y_pred_rf.append(rf.predict_one(Xi))
    y_score_rf.append(rf.predict_proba_one(Xi)[True])

    y_pred_dt.append(dt.predict_one(Xi))
    y_score_dt.append(dt.predict_proba_one(Xi)[True])

    y_pred_lr.append(lr.predict_one(Xi))
    y_score_lr.append(lr.predict_proba_one(Xi)[True])

    y_pred_pac.append(pac.predict_one(Xi))
    y_score_pac.append(pac.predict_proba_one(Xi)[True])

    y_pred_knn.append(knn.predict_one(Xi))
    y_score_knn.append(knn.predict_proba_one(Xi)[True])

    y_true.append(yi)

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_true.npy', np.asarray(y_true))

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_pred_rf.npy', np.asarray(y_pred_rf))
np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_score_rf.npy', np.asarray(y_score_rf))

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_pred_dt.npy', np.asarray(y_pred_dt))
np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_score_dt.npy', np.asarray(y_score_dt))

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_pred_lr.npy', np.asarray(y_pred_lr))
np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_score_lr.npy', np.asarray(y_score_lr))

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_pred_pac.npy', np.asarray(y_pred_pac))
np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_score_pac.npy', np.asarray(y_score_pac))

np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_pred_knn.npy', np.asarray(y_pred_knn))
np.save('./temp/model' + dit_str[model_type] + '_data' + dit_str[data_type] + '_' + str(n_days_lookahead) + '_days_lookahead_y_score_knn.npy', np.asarray(y_score_knn))
