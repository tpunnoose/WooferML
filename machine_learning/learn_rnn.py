from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h = nn.Sequential(
                  nn.Linear(input_size + hidden_size, 50),
                  nn.Softsign(),
                  nn.Linear(50, 15),
                  nn.Softsign(),
                  nn.Linear(15, hidden_size)
                )
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2o = nn.Sequential(
                  nn.Linear(input_size + hidden_size, 50),
                  nn.Softsign(),
                  nn.Linear(50, 15),
                  nn.Softsign(),
                  nn.Linear(15, output_size)
                )

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class FreeTorqueDataset(Dataset):
    """Robot dataset to calculate free torque."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with data.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.len = len(self.data_frame.index)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        joint_history = np.array(self.data_frame.iloc[idx, 0:6]).astype('float')
        torques = np.array(self.data_frame.iloc[idx, 6:9]).astype('float')

        sample = (joint_history, torques)

        return sample

def main():
    train_dataset = FreeTorqueDataset('rnn_train.csv')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                              shuffle=False, num_workers=1)

    model = RNN(6, 3, 3)

    # number of steps to keep forward propogation
    k1 = 1000

    # number of steps to keep backward propogation
    k2 = 50

    mse = nn.MSELoss()

    learning_rate = 0.001

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i % k1 == 0:
                hidden = model.initHidden()

                joints, torque = data
                torque_estimated, hidden = model(joints.float(), hidden)
            else:
                joints, torque = data
                torque_estimated, hidden = model(joints.float(), hidden)

                if i % k2 == 0:
                    loss = mse(torque_estimated, torque.float())
                    loss.backward(retain_graph=True)

                    # Add parameters' gradients to their values, multiplied by learning rate
                    for p in model.parameters():
                        p.data.add_(-learning_rate, p.grad.data)

                    model.zero_grad()

                    # print statistics
                    running_loss += loss.item()
                    if i % k2*20 == 0:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 1000))
                        running_loss = 0.0

    print('Finished Training')

    PATH = './rnn_net.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    main()
