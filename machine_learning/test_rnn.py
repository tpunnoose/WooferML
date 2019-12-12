from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from learn_rnn import FreeTorqueDataset, RNN

PATH = './rnn_net.pth'
model = RNN(6, 3, 3)

model.load_state_dict(torch.load(PATH))
model.eval()

test_dataset = FreeTorqueDataset('rnn_test.csv')
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=False, num_workers=1)

mse = nn.MSELoss()

k1 = 100

loss = 0

for i, data in enumerate(testloader, 0):
    joints, torque = data

    if i % k1 == 0:
        hidden = model.initHidden()

    torque_estimated, hidden = model(joints.float(), hidden)

    loss += mse(torque_estimated, torque.float())

print(loss/(i/k1))
