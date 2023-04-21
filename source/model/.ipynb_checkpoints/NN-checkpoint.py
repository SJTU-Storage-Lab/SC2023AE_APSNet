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
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_size, output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x
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