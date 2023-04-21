import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn import metrics

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, bidirectional=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = len(x)
        if self.bidirectional:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _= self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        out = self.sigmoid(out)
        return out
    
    pass

class DatasetUtil(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
    pass