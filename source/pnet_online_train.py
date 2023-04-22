import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.PNet import LSTM, NN, PSiameseNetwork, DatasetUtil, ContrastiveLoss
import joblib
from utils import model_and_dataset_selection

n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict = model_and_dataset_selection.train_select_online()


def loadData():

    if data_folder_name_dict[data_type] != 'C':
        X_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', allow_pickle=True)
        y_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', allow_pickle=True)
    else:
        X_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train_pnet.npy', allow_pickle=True)
        y_train = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels_pnet.npy', allow_pickle=True)
    aging = np.load('../data/aging.npy')
    for i in range(0, len(y_train)):
        if y_train[i] == 0:
            y_train[i] = 1
        else:
            y_train[i] = 0
    X_test = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', allow_pickle=True)
    y_test = np.load('../data/' + data_folder_name_dict[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', allow_pickle=True)
    X_train = X_train.astype('float32')
    aging = aging.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    return X_train, aging, y_train, X_test, y_test


def print_all_metrics(true, predicted):
    print(metrics.classification_report(true, predicted, digits=5))


def decision(x_smart, aging, model):
    with torch.no_grad():
        out1, out2 = model(x_smart, aging)
        distance = F.pairwise_distance(out1, out2, p=2)
    return distance


# ----------- Hyperparameters -----------#
num_epochs = 3000 # 5, 1500
batch_size = 512
lr = 0.0002
input_size_smart = 330
hidden_size_smart = 128
output_size_smart = 48
input_size_aging = 5
hidden_size_aging = 32
output_size_aging = 48
output_size_model = 64
num_layers_smart = 3
num_layers_aging = 3
similarity_limit = 0.48
# out 16 hidden 12 ratio 0.2 votingnum 100 limit 1.168
# 12 16 0.75 100 1.205
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pnet = PSiameseNetwork(input_size_smart=input_size_smart, input_size_aging=input_size_aging,
                       hidden_size_smart=hidden_size_smart, hidden_size_aging=hidden_size_aging,
                       num_layers_smart=num_layers_smart, num_layers_aging=num_layers_aging,
                       output_size_smart=output_size_smart, output_size_aging=output_size_aging,
                       output_size_model=output_size_model)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(pnet.parameters(), lr=lr)
torch.save(pnet.state_dict(), '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_online.pth')
torch.save(optimizer.state_dict(), '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_optimizer_online.pth')

X_train_smart, aging, y_train, X_test_smart, y_test = loadData()

print(y_train)
print(X_train_smart.shape, aging.shape, y_train.shape, X_test_smart.shape, y_test.shape)

train_dataset = DatasetUtil(X_train_smart, aging, y_train)
test_dataset = DatasetUtil(X_test_smart, aging, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#------------------- Train -------------------#
Loss_list = []
counter = []
index = 0
# classification_report_list = []
# auc_list = []
for epoch in range(num_epochs):
    print('training', epoch)
    for i, (X, aging, y) in enumerate(train_loader):
        pnet = PSiameseNetwork(input_size_smart=input_size_smart, input_size_aging=input_size_aging,
                               hidden_size_smart=hidden_size_smart, hidden_size_aging=hidden_size_aging,
                               num_layers_smart=num_layers_smart, num_layers_aging=num_layers_aging,
                               output_size_smart=output_size_smart, output_size_aging=output_size_aging,
                               output_size_model=output_size_model)
        optimizer = torch.optim.Adam(pnet.parameters(), lr=lr)
        pnet.load_state_dict(torch.load('../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_online.pth'))
        optimizer.load_state_dict(torch.load('../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_optimizer_online.pth'))

        X = X.to(device)
        aging = aging.to(device)
        y = y.to(device)

        out1, out2 = pnet(X, aging)

        optimizer.zero_grad()
        loss_contrastive = criterion(out1, out2, y)
        loss_contrastive.backward()
        optimizer.step()

        Loss_list.append(loss_contrastive.item())

        torch.save(pnet.state_dict(), '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_online.pth')
        torch.save(optimizer.state_dict(), '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_optimizer_online.pth')
        pass

    # ----------- PNet -----------
    if epoch % 50 == 0:
        print('testing', epoch)
        y_predicted = []
        y_true = []
        with torch.no_grad():
            for i, (X, aging, y) in enumerate(test_loader):
                X = X.to(device)
                aging = aging.to(device)
                y = y.to(device)
                distance = decision(X, aging, pnet)

                for j in range(0, len(distance)):
                    if distance[j] > similarity_limit:
                        y_pred = 0
                    else:
                        y_pred = 1
                    y_predicted.append(y_pred)
                    y_true.append(y[j].cpu())

        print_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
        torch.save(pnet.state_dict(), '../trained_model/' + model_folder_name_dict[model_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_checkpoint/pnet_online_'+str(epoch)+'.pth')

print('final testing')
y_predicted = []
y_true = []
with torch.no_grad():
    for i, (X, aging, y) in enumerate(test_loader):
        X = X.to(device)
        aging = aging.to(device)
        y = y.to(device)
        distance = decision(X, aging, pnet)

        for j in range(0, len(distance)):
            if distance[j] > similarity_limit:
                y_pred = 0
            else:
                y_pred = 1
            y_predicted.append(y_pred)
            y_true.append(y[j].cpu())
    print_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
    print('Done')
