import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
PATH = 'model.pt'
torch.save(net.state_dict(), PATH)

device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))

torch.save(net.state_dict(), PATH)

device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)

torch.save(net.state_dict(), PATH)

device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)
