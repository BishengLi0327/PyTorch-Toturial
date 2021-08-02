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


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Save and load the model via state_dict
PATH = 'state_dict_model.pt'
torch.save(net.state_dict(), PATH)

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

# Save and load the entire model
PATH = 'entire_model.pt'
torch.save(net, PATH)

model = torch.load(PATH)
model.eval()
