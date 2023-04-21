from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

n = {5: 10000, 7: 11000, 15: 12000, 30: 13000, 45: 14000, 60: 15000, 90: 17500, 120: 25000}

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2}: '))

if(data_type not in ['A', 'B', 'C']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'mc1_no_aging_attr', 'E': 'balanced_mc1_mc2', 'F': '500_mc2'}


def loadData():

    X_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', allow_pickle=True)
    y_train = np.load('../data/' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', allow_pickle=True)
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


def search_svm(X_train, y_train, X_test, y_test):
    print('---------------- SVM ----------------')
    parameters = [
        {'kernel': ['rbf', 'poly'], 'gamma': [0.5, 0.1, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    ]

    grid_clf = GridSearchCV(SVC(), param_grid=parameters, cv=5)
    grid_clf.fit(X_train, y_train)
    print('最优分类器:', grid_clf.best_params_, '最优分数:', grid_clf.best_score_)

    print('测试集结果：')
    y_true, y_pred = y_test, grid_clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    pass


def search_dt(X_train, y_train, X_test, y_test):
    print('---------------- DT ----------------')
    parameters = [
        {'criterion': ['gini', 'entropy'],
         'max_depth':[None, 10, 30, 50, 80, 100],
         'min_samples_leaf':[1, 2, 3, 5, 10],
         'min_impurity_decrease':[0.0, 0.1, 0.2, 0.5],
         'max_leaf_nodes':[None, 10, 30, 50, 60, 100]
         },
    ]

    grid_clf = GridSearchCV(DTC(), param_grid=parameters, cv=5)
    grid_clf.fit(X_train, y_train)
    print('最优分类器:', grid_clf.best_params_, '最优分数:', grid_clf.best_score_)

    print('测试集结果：')
    y_true, y_pred = y_test, grid_clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    pass


def search_knn(X_train, y_train, X_test, y_test):
    print('---------------- KNN ----------------')
    parameters = [
        {'weights': ['uniform', 'distance'], 'n_neighbors':[i for i in range(1, 11)], 'p':[1, 2]}
    ]

    grid_clf = GridSearchCV(KNC(), param_grid=parameters, cv=5)
    grid_clf.fit(X_train, y_train)
    print('最优分类器:', grid_clf.best_params_, '最优分数:', grid_clf.best_score_)

    print('测试集结果：')
    y_true, y_pred = y_test, grid_clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    pass


def search_lr(X_train, y_train, X_test, y_test):
    print('---------------- LR ----------------')
    parameters = [
        {
            'C': [0.1, 1, 10],
            'max_iter':[10, 50, 100, 150, 200],
            'penalty':['l2', 'none'],
            'solver':['newton-cg', 'lbfgs']
        },
        {
            'C': [0.1, 1, 10],
            'max_iter':[10, 50, 100, 150, 200],
            'penalty':['l1', 'l2'],
            'solver':['liblinear']
        }
    ]
    grid_clf = GridSearchCV(LR(), param_grid=parameters, cv=5)
    grid_clf.fit(X_train, y_train)
    print('最优分类器:', grid_clf.best_params_, '最优分数:', grid_clf.best_score_)

    print('测试集结果：')
    y_true, y_pred = y_test, grid_clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    pass


def search_rf(X_train, y_train, X_test, y_test):
    print('---------------- RF ----------------')
    parameters = [
        {
            'n_estimators': [10, 50, 100, 150, 200],
            'criterion': ["gini", "entropy"],
            'max_depth': [None, 10, 100],
            'min_samples_split':[0.5, 2, 3, 5],
            'min_samples_leaf':[1, 2, 3, 5],
            'max_leaf_nodes':[None, 10, 100]
        }
    ]

    grid_clf = GridSearchCV(RFC(), param_grid=parameters, cv=5)
    grid_clf.fit(X_train, y_train)
    print('最优分类器:', grid_clf.best_params_, '最优分数:', grid_clf.best_score_)

    print('测试集结果：')
    y_true, y_pred = y_test, grid_clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    pass


def main():

    X_train, y_train, X_test, y_test = loadData()
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))

#     search_lr(X_train, y_train, X_test, y_test)
    search_rf(X_train, y_train, X_test, y_test)
    search_dt(X_train, y_train, X_test, y_test)
    search_svm(X_train, y_train, X_test, y_test)
    search_knn(X_train, y_train, X_test, y_test)

    pass


if __name__ == '__main__':
    main()
    pass
