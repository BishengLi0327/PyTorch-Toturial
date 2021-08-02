import torch
import torchvision
from torch import nn
import torch.optim as optim

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)

loss = (prediction - labels).sum()
print(loss)
loss.backward()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer.step()


model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)
# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
