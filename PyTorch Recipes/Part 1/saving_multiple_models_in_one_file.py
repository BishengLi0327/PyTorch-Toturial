import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


netA = Net()
netB = Net()

optimizerA = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
optimizerB = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)

PATH = 'several_model.pt'
torch.save({
            "modelA's state_dict": netA.state_dict(),
            "modelB's state_dict": netB.state_dict(),
            "optimizerA's state_dict": optimizerA.state_dict(),
            "optimizerB's state_dict": optimizerB.state_dict()
            }, PATH)


modelA = Net()
modelB = Net()
optimModelA = optim.SGD(modelA.parameters(), lr=0.001, momentum=0.9)
optimModelB = optim.SGD(modelB.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA\'s state_dict'])
modelB.load_state_dict(checkpoint['modelB\'s state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA\'s state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB\'s state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
