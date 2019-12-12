from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


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

        joint_history = np.array(self.data_frame.iloc[idx, 0:18]).astype('float')
        torques = np.array(self.data_frame.iloc[idx, 18:21]).astype('float')

        sample = (joint_history, torques)

        return sample

def main():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    train_dataset = FreeTorqueDataset('train.csv')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50,
                                              shuffle=True, num_workers=1)



    # model = nn.Linear(6,3)
    model = nn.Sequential(
              nn.Linear(18, 100),
              nn.Sigmoid(),
              nn.Linear(100, 50),
              nn.Sigmoid(),
              nn.Linear(50, 3)
            )

    mse = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #
    #
    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            joints, torque = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            torque_estimated = model(joints.float())
            loss = mse(torque_estimated, torque.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './free_torque_net.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    main()
