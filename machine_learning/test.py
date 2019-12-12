from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from learn import FreeTorqueDataset

PATH = './free_torque_net.pth'
model = nn.Sequential(
          nn.Linear(18, 100),
          nn.Dropout(0.5),
          nn.Softsign(),
          nn.Linear(100, 50),
          nn.Dropout(0.5),
          nn.Softsign(),
          nn.Linear(50, 3)
        )

model.load_state_dict(torch.load(PATH))
model.eval()

test_dataset = FreeTorqueDataset('test.csv')
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset),
                                          shuffle=True, num_workers=1)

mse = nn.MSELoss()

for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    joints, torque = data

    # forward + backward + optimize
    torque_estimated = model(joints.float())
    loss = mse(torque_estimated, torque.float())

    print(loss)
