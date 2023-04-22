import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class SiameseNetwork(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 output_size_net,
                 output_size_model):
        super(SiameseNetwork, self).__init__()
        
        self.lstm = LSTM(input_size, hidden_size, num_layers, output_size_net).to(device)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(output_size_net, output_size_model).to(device)
    
    def forward(self, x1, x2):
        out1 = self.lstm(x1)
        out1 = self.relu(out1)
        out1 = self.fc(out1)
        
        out2 = self.lstm(x2)
        out2 = self.relu(out2)
        out2 = self.fc(out2)
        
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
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        return self.x[index][0], self.x[index][1], self.y[index]
    def __len__(self):
        return len(self.y)
    
    pass