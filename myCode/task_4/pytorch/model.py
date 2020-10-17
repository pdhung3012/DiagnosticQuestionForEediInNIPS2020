import torch
import torch.nn as nn
import numpy as np

class PyTorchModel(nn.Module):
    """
    Simple example model to illustrate saving and loading.
    """
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(10,20)
        self.fc2 = nn.Linear(20,10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def predict(self, masked_data, masked_binary_data):
        predictions = np.zeros_like(masked_binary_data)
        predictions = np.random.choice([0,1], size=masked_binary_data.shape)
        return predictions

    def select_feature(self, masked_data, can_query):
        selections = []
        for i in range(masked_data.shape[0]):
            can_query_row = can_query[i,:]
            selected_feature = np.random.choice(np.where(can_query_row==1)[0])
            selections.append(selected_feature)
        return selections