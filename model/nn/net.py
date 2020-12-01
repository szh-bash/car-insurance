import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.drop_out1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.drop_out2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_out1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out2(x)
        x = self.fc3(x)
        return x
