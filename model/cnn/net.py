import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 2560)
        self.fc3 = nn.Linear(2560, 5120)
        self.fc4 = nn.Linear(5120, 10240)
        self.fc5 = nn.Linear(10240, 10240)
        self.fc6 = nn.Linear(10240, 5120)
        self.fc7 = nn.Linear(5120, 2560)
        self.fc8 = nn.Linear(2560, 512)
        self.fc9 = nn.Linear(512, 2)
        # self.drop_out1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
