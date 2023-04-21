import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from model.SN import LSTM, SiameseNetwork, DatasetUtil, ContrastiveLoss
import joblib

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

def loadData():

    X_train = np.load('../data/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/smart_sn_train.npy',allow_pickle=True)
    y_train = np.load('../data/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/labels_sn_train.npy',allow_pickle=True)
    X_test = np.load('../data/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/smart_sn_test.npy',allow_pickle=True)
    y_test = np.load('../data/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/labels_sn_test.npy',allow_pickle=True)
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(X_test)
    np.random.set_state(state)
    np.random.shuffle(y_test)

    return X_train, y_train, X_test, y_test

def print_all_metrics(true, predicted):
    print(metrics.classification_report(true, predicted, digits=5))
    
def decision(X1, X2, model):
    with torch.no_grad():
        out1, out2 = model(X1, X2)
        distance = F.pairwise_distance(out1, out2, p=2)
    return distance
    
### ----------- Hyperparameters -----------#
num_epochs = 200 # 5, 1500
batch_size = 512
lr = 0.0002
input_size = 11
hidden_size =  24
output_size_net = 32
output_size_model = 32
num_layers = 3

similarity_limit = 0.48

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

sn = SiameseNetwork(input_size, hidden_size, num_layers, output_size_net, output_size_model)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(sn.parameters(), lr=lr)

X_train, y_train, X_test, y_test = loadData()

print(y_train)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_dataset = DatasetUtil(X_train, y_train)
test_dataset = DatasetUtil(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#------------------- Train -------------------#
Loss_list = []
counter = []
index = 0
for epoch in range(num_epochs):
    print('training', epoch)
    for i, (X1, X2, y) in enumerate(train_loader):   
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        
        out1, out2 = sn(X1, X2)
        
        optimizer.zero_grad()
        loss_contrastive = criterion(out1, out2, y)
        loss_contrastive.backward()
        optimizer.step()

        Loss_list.append(loss_contrastive.item())
        pass
    
    # ----------- SN -----------
    if epoch%50 == 0:
        print('testing', epoch)
        y_predicted = []
        y_true = []
        with torch.no_grad():
            for i, (X1, X2, y) in enumerate(train_loader):
                X1 = X1.to(device)
                X2 = X2.to(device)
                y = y.to(device)
                distance = decision(X1, X2, sn)
                
                for j in range(0,len(distance)):
                    if distance[j] > similarity_limit:
                        y_pred = 0
                    else:
                        y_pred = 1           
                    y_predicted.append(y_pred)
                    y_true.append(y[j].cpu())
                    
        print_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
        torch.save(sn.state_dict(), '../trained_model/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/sn_checkpoint/sn_' + str(epoch) + '.pth')
           
print('final testing')
y_predicted = []
y_true = []
with torch.no_grad():
    for i, (X1, X2, y) in enumerate(test_loader):
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        distance = decision(X1, X2, sn)
        print(y)
        print(distance)

        for j in range(0,len(distance)):
            if distance[j] > similarity_limit:
                y_pred = 0
            else:
                y_pred = 1           
            y_predicted.append(y_pred)
            y_true.append(y[j].cpu())
            
    print_all_metrics(np.asarray(y_true).astype('int'), np.asarray(y_predicted))
    torch.save(sn.state_dict(), '../trained_model/sn_mc1_mc2/' + str(n_days_lookahead) + '_days_lookahead/sn.pth')
    print('Done')
