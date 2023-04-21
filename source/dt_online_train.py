import numpy as np
import joblib
import pickle
from sklearn import metrics
from river.tree import HoeffdingAdaptiveTreeClassifier

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2}: '))

if(data_type not in ['A', 'B', 'C']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2'}


def loadData():

    X_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', allow_pickle=True)
    y_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', allow_pickle=True)
    X_test = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', allow_pickle=True)
    y_test = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', allow_pickle=True)
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    np.random.set_state(state)
    np.random.shuffle(X_test)
    np.random.set_state(state)
    np.random.shuffle(y_test)

    return X_train, y_train, X_test, y_test


def print_all_metrics(true, predicted):
    print(metrics.classification_report(true, predicted, digits=5))


X_train, y_train, X_test, y_test = loadData()
print('------------------ Loading Data ------------------')
X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))
print(X_train.shape)

headers = [str(i) for i in range(330)]
print(headers)
data_x = [dict(zip(headers, x)) for x in X_train]
data_y = [True if y == 1 else False for y in y_train]

model = HoeffdingAdaptiveTreeClassifier(grace_period=200, leaf_prediction='nba', split_criterion='info_gain')
i = 0
for (Xi, yi) in zip(data_x, data_y):
    if i % 1000 == 0:
        print(i)
    i += 1
    model.learn_one(Xi, yi)

headers = [str(i) for i in range(330)]
data_x_test = [dict(zip(headers, x)) for x in X_test]
data_y_test = [True if y == 1 else False for y in y_test]
y_true = []
y_pred = []
for Xi, yi in zip(data_x_test, data_y_test):
    yi_pred = model.predict_one(Xi)
    y_true.append(yi)
    y_pred.append(yi_pred)
print_all_metrics(np.asarray(y_test), np.asarray(y_pred))

joblib.dump(model, '../trained_model/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/dt_online.pkl')
