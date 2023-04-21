import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import KFold
import pandas as pd    
import numpy as np
from sklearn.preprocessing import normalize

from model_source.PNet import LSTM, NN, PSiameseNetwork, DatasetUtil, ContrastiveLoss

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

n = {5:12500,7:12500,15:12500,30:12500,45:15000,60:18000,90:20000,120:30000}

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Manufacturer 1 without aging information, E - Balanced Manufacturer 1 & 2}: '))

if(data_type not in ['A', 'B', 'C', 'D', 'E']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A':'mc1', 'B':'mc2', 'C':'mc1_mc2', 'D':'mc1_no_aging_attr', 'E':'balanced_mc1_mc2'}

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
#         if self.bidirectional:
#             self.fc = nn.Linear(hidden_size*2, output_size)
#         else:
#             self.fc = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         batch_size = len(x)
#         if self.bidirectional:
#             h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
#             c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
#         else:
#             h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
#             c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
#         x, _= self.lstm(x, (h0,c0))
#         x = self.relu(x)
#         x = self.fc(x[:,-1,:])
#         return x
    
#     pass

# class NN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super(NN, self).__init__()
#         self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True))
#         self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(True))
#         self.layer3 = nn.Sequential(nn.Linear(hidden_size, output_size))

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x
#     pass

# class PSiameseNetwork(nn.Module):
#     def __init__(self,\
#                  input_size_smart, input_size_aging,\
#                  hidden_size_smart, hidden_size_aging,\
#                  num_layers_smart, num_layers_aging,\
#                  output_size_smart, output_size_aging, output_size_model):
#         super(PSiameseNetwork, self).__init__()
        
#         # smart
# #         self.network_smart = LSTM(input_size=input_size_smart, hidden_size=hidden_size_smart,\
# #                                   num_layers=num_layers_smart, output_size=output_size_smart).to(device)

#         self.network_smart = NN(150, 256, output_size_smart).to(device)
#         self.relu_smart_1 = nn.ReLU()
#         self.fc_smart = nn.Linear(output_size_smart, output_size_model).to(device)
        
#         # aging
#         self.network_aging = LSTM(input_size=input_size_aging, hidden_size=hidden_size_aging,\
#                                   num_layers=num_layers_aging, output_size=output_size_aging).to(device)
# #         self.network_aging = NN(70, 128, output_size_aging).to(device)
#         self.relu_aging_1 = nn.ReLU()
#         self.fc_aging = nn.Linear(output_size_aging, output_size_model).to(device)
    
#     def forward(self, x_smart, x_aging):
#         x_smart = x_smart.reshape((len(x_smart), -1))
#         out1 = self.network_smart(x_smart)
#         out1 = self.relu_smart_1(out1)
#         out1 = self.fc_smart(out1)
        
# #         x_aging = x_aging.reshape((len(x_aging), -1))
#         out2 = self.network_aging(x_aging)
#         out2 = self.relu_aging_1(out2)
#         out2 = self.fc_aging(out2)
        
#         return out1, out2
    
#     pass

# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +\
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
#         return loss_contrastive*0.5
    
# class DatasetUtil(Dataset):
#     def __init__(self, x, aging, y):
#         self.x = x
#         self.aging = aging
#         self.y = y
#     def __getitem__(self, index):
#         slice_=np.random.choice(self.aging.shape[0],10)
#         return self.x[index], np.mean(self.aging[slice_], axis=0), self.y[index]
#     def __len__(self):
#         return len(self.y)
    
#     pass


def loadData():

    X = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy',allow_pickle=True)
    y = np.load('../data/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy',allow_pickle=True)
    aging = np.load('../data/aging.npy', allow_pickle = True)
    
    
    X = X.astype('float32')
    y = y.astype('float32')
    aging = aging.astype('float32')
    return X, y, aging

def print_all_metrics(true, predicted, score=None):
    print(metrics.classification_report(true, predicted, digits=5))
    print('accuracy', metrics.accuracy_score(true, predicted))
    if(score is not None):
        print('auc', metrics.roc_auc_score(true, score))
    
def decision(x_smart, aging, model):
    with torch.no_grad():
        out1, out2 = model(x_smart, aging)
        distance = F.pairwise_distance(out1, out2, p=2)
    return distance

def computeScore(distance, limit):
    if distance>1:
        return 0.01
    elif distance<limit:
        return float(1-distance/limit*0.5)
    else: 
        # distance>=limit
        return float(1-(0.5+(distance-limit)/(1-limit)*0.5))
    
def saveExcel(data, path):
    data_df = pd.DataFrame(data) 
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer,'page_1') 
    writer.save()
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity_limit = 0.55
print(device)

pnet = torch.load('../trained_model/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/pnet.pkl').to(device)
rf = joblib.load('../trained_model/' + dir_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/rf.pkl')

X_test, y_test, aging = loadData()

test_dataset = DatasetUtil(X_test, aging, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

print('------------------ APSNet ------------------')
# 1: 0.71, 0: 0.42

best_upper_limit_dit = {5:0.75,7:0.77,15:0.72,30:0.70,45:0.70,60:0.75,90:0.70,120:0.68}
best_lower_limit_dit = {5:0.42,7:0.45,15:0.41,30:0.42,45:0.40,60:0.40,90:0.40,120:0.35}

upper_limit = best_upper_limit_dit[n_days_lookahead]
lower_limit = best_lower_limit_dit[n_days_lookahead]

y_pred_union_list = []
y_pred_rf_list = []
y_score_list = []
y_true_list = []
with torch.no_grad():
    for i, (X_apsnet, aging, y_apsnet) in enumerate(test_loader):
        y_pred_rf = rf.predict(X_apsnet.reshape((-1,330)))
        distance = decision(X_apsnet.to(device), aging.to(device), pnet)

        for j in range(0,len(distance)):
            if y_pred_rf[j] == 1:
                score = computeScore(distance[j], upper_limit)
                y_score_list.append(score)
                y_pred_union_list.append(1 if distance[j]<=upper_limit else 0)
            else:
                score = computeScore(distance[j], lower_limit)
                y_score_list.append(score)
                y_pred_union_list.append(1 if distance[j]<=lower_limit else 0)                
            y_true_list.append(y_apsnet[j])
            y_pred_rf_list.append(y_pred_rf[j])

print_all_metrics(np.asarray(y_true_list).astype('int'), np.asarray(y_pred_union_list), np.asarray(y_score_list))
