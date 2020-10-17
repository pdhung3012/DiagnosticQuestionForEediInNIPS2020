'''
a dummy script to "train" a model and save the "trained model"
(just saved an initialized model for local testing) 
'''

import os
import torch
from model import PyTorchModel

model = PyTorchModel()
print("Model params in train_model.py:")
for param in model.state_dict():
    print(param, "\t", model.state_dict()[param].size())

torch.save(model.state_dict(), 'model_task_4.pt')