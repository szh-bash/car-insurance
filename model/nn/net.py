import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(159, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_out1 = nn.Dropout(p=0.2)
        self.drop_out2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop_out1(x)
        x = F.relu(self.fc2(x))
        # x = self.drop_out1(x)
        x = torch.sigmoid(self.fc3(x))
        x = x.view(-1)
        return x
