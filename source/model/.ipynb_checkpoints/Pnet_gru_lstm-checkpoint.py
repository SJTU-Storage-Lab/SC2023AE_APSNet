import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = len(x)
        x, _= self.gru(x)
        x = self.relu(x)
        x = self.fc(x[:,-1,:])
        return x
    
    pass

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = len(x)
        if self.bidirectional:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        x, _= self.lstm(x, (h0,c0))
        x = self.relu(x)
        x = self.fc(x[:,-1,:])
        return x
    
    pass

class PSiameseNetwork(nn.Module):
    def __init__(self,\
                 input_size_smart, input_size_aging,\
                 hidden_size_smart, hidden_size_aging,\
                 num_layers_smart, num_layers_aging,\
                 output_size_smart, output_size_aging, output_size_model):
        super(PSiameseNetwork, self).__init__()
        
        # smart
        self.network_smart = GRU(input_size=input_size_smart, hidden_size=hidden_size_smart,\
                                  num_layers=num_layers_smart, output_size=output_size_smart).to(device)

#         self.network_smart = NN(330, 512, output_size_smart).to(device)
        self.relu_smart_1 = nn.ReLU()
        self.fc_smart = nn.Linear(output_size_smart, output_size_model).to(device)
        
        # aging
        self.network_aging = LSTM(input_size=input_size_aging, hidden_size=hidden_size_aging,\
                                  num_layers=num_layers_aging, output_size=output_size_aging).to(device)
#         self.network_aging = NN(70, 128, output_size_aging).to(device)
        self.relu_aging_1 = nn.ReLU()
        self.fc_aging = nn.Linear(output_size_aging, output_size_model).to(device)
    
    def forward(self, x_smart, x_aging):
#         x_smart = x_smart.reshape((len(x_smart), -1))
        out1 = self.network_smart(x_smart)
        out1 = self.relu_smart_1(out1)
        out1 = self.fc_smart(out1)
        
#         x_aging = x_aging.reshape((len(x_aging), -1))
        out2 = self.network_aging(x_aging)
        out2 = self.relu_aging_1(out2)
        out2 = self.fc_aging(out2)
        
        return out1, out2
    
    pass

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +\
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive*0.5
    
class DatasetUtil(Dataset):
    def __init__(self, x1, aging, y):
        self.x1 = x1
        self.aging = aging
        self.y = y
    def __getitem__(self, index):
        slice=np.random.choice(aging.shape[0],10)
        return self.x1[index], np.mean(self.aging[slice], axis=0), self.y[index]
    def __len__(self):
        return len(self.y)
