import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics

from model_source.PNet_gru_lstm import GRU, LSTM, PSiameseNetwork, DatasetUtil, ContrastiveLoss

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Manufacturer 1 without aging information, E - Balanced Manufacturer 1 & 2}: '))

if(data_type not in ['A', 'B', 'C', 'D', 'E']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A':'mc1', 'B':'mc2', 'C':'mc1_mc2', 'D':'mc1_no_aging_attr', 'E':'balanced_mc1_mc2'}

def loadData():

    X_train = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy',allow_pickle=True)
    aging = np.load('../data/aging.npy')
    y_train = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy',allow_pickle=True)
    for i in range(0,len(y_train)):
        if y_train[i] == 0:
            y_train[i] = 1
        else:
            y_train[i] = 0
    X_test = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy',allow_pickle=True)
    y_test = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy',allow_pickle=True)
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

def get_all_metrics(true, predicted):
    print(metrics.classification_report(true, predicted, digits=5))
    print('accuracy', metrics.accuracy_score(true, predicted))
    print('auc', metrics.roc_auc_score(true, predicted))
    
def decision(x_smart, aging, model):
    with torch.no_grad():
        out1, out2 = model(x_smart, aging)
        distance = F.pairwise_distance(out1, out2, p=2)
    return distance
    
### ----------- Hyperparameters -----------#
num_epochs = 1500
batch_size = 64
lr = 0.0001
input_size_smart = 11
input_size_aging = 5
output_size_smart = 48
output_size_aging = 48
output_size_model = 64
hidden_size_smart = 32
hidden_size_aging = 32
num_layers_smart = 3
num_layers_aging = 3
# out 16 hidden 12 ratio 0.2 votingnum 100 limit 1.168
# 12 16 0.75 100 1.205
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = PSiameseNetwork(input_size_smart=input_size_smart, input_size_aging=input_size_aging,\
                      hidden_size_smart=hidden_size_smart, hidden_size_aging=hidden_size_aging,\
                      num_layers_smart=num_layers_smart, num_layers_aging=num_layers_aging,\
                      output_size_smart=output_size_smart, output_size_aging=output_size_aging,\
                      output_size_model=output_size_model)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

X_train, aging, y_train, X_test, y_test = loadData()

train_dataset = DatasetUtil(X_train, aging, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DatasetUtil(X_test, aging, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#------------------- Train -------------------#
Loss_list = []
counter = []
index = 0
# classification_report_list = []
# auc_list = []
for epoch in range(num_epochs):
    print('training', epoch)
    for i, (X_smart, X_aging, y) in enumerate(train_loader):
        X_smart = X_smart.to(device)
        X_aging = X_aging.to(device)
        y = y.to(device)
#         print(X_smart.shape, X_aging.shape)
        out1, out2 = model(X_smart,X_aging)
        optimizer.zero_grad()
        loss_contrastive = criterion(out1, out2, y)
        loss_contrastive.backward()
        optimizer.step()

        Loss_list.append(loss_contrastive.item())
        pass
    
    if epoch%50 == 0:
        print('testing', epoch)
        y_predicted = []
        y_true = []
        with torch.no_grad():
            for i, (X_smart, X_aging, y) in enumerate(test_loader):
                X_smart = X_smart.to(device)
                X_aging = X_aging.to(device)
                y = y.to(device)
                distance = decision(X_smart.to(device), X_aging.to(device), model)
                for j in range(0,len(distance)):
                    y_predicted.append(1 if distance[j]<=0.48 else 0)
                    y_true.append(y[j].cpu())
        get_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
#         np.save('./loss/loss_pnet_m1_m2_0_'+str(epoch)+'.npy',Loss_list)
#         torch.save(model, './model/model_pnet_2gru_m1_m2_0_'+str(epoch)+'.pkl')
        
print('final testing')
y_predicted = []
y_true = []
with torch.no_grad():
    for i, (X_smart, X_aging, y) in enumerate(test_loader):
        X_smart = X_smart.to(device)
        X_aging = X_aging.to(device)
        y = y.to(device)
        distance = decision(X_smart.to(device), X_aging.to(device), model)
        for j in range(0,len(distance)):
            y_predicted.append(1 if distance[j]<=0.48 else 0)
            y_true.append(y[j].cpu()) 
    get_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
#     np.save('./loss/loss_pnet_m1_m2_0_final.npy',Loss_list)
    torch.save(model, '../trained_model/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet_gru_lstm.pkl')
    print('Done')
