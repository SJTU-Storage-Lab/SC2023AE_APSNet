from river import linear_model
from river import optim
from river import utils
from river.tree import HoeffdingAdaptiveTreeClassifier
from river import model_selection
import numpy as np
from river import metrics

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
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')

    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    return X_train.reshape((len(X_train), -1)), y_train


def grid_search_dt(X_train, y_train):
    print('------------------- Decision Tree -------------------')
    dt = HoeffdingAdaptiveTreeClassifier()
    grid_dt = {
        'grace_period ': [150, 200, 250],
        #     'max_depth': [None, 10, 100],
        #     'split_criterion': ['gini', 'info_gain', 'hellinger'],
        #     'leaf_prediction': ['mc', 'nb', 'nba']
    }
    dts = utils.expand_param_grid(dt, grid_dt)
    print(len(dts))
    sh_dt = model_selection.SuccessiveHalvingClassifier(
        dts,
        metric=metrics.Accuracy(),
        budget=2000,
        eta=2,
        verbose=True
    )

    headers = [str(i) for i in range(330)]
    print(headers)
    data_x = [dict(zip(headers, x)) for x in X_train]
    data_y = [True if y == 1 else False for y in y_train]

    i = 0
    for (Xi, yi) in zip(data_x, data_y):
        if i % 1000 == 0:
            print(i)
        i += 1
        sh_dt.learn_one(Xi, yi)

    print(sh_dt.best_model)


def main():
    X_train, y_train = loadData()
    grid_search_dt(X_train, y_train)


if __name__ == '__main__':
    main()
