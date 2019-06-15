import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

transform1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform2= transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.2),
                 transforms.RandomRotation(degrees=8),
                 transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                       download=True,transform=transform2)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                       download=True,transform=transform1)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=True,num_workers=2)





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batch = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # conv1
        x = self.pool(F.relu(self.conv2(x)))  #conv2
        x = self.drop(x)
        x = F.relu(self.batch(self.conv2(x))) #conv3
        x = self.pool(F.relu(self.conv2(x)))  #conv4
        x = self.drop(x)
        x = F.relu(self.batch(self.conv2(x))) #conv5
        x = self.drop(F.relu(self.conv3(x)))  #conv6
        x = F.relu(self.batch(self.conv3(x))) #conv7
        x = F.relu(self.batch(self.conv3(x))) #conv8
        x = self.drop(x)
        # Fully connected
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(50):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if(epoch>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()
    print('Epoch ',epoch,' Finished Training')

# TEST
correct = 0
total = 0
# with torch.no_grad():
for data in testloader:
    images, labels = data
    images, labels_cuda = Variable(images.cuda()),Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_cuda.size(0)
    predicted_np = predicted.cpu().numpy()
    labels_np = labels.numpy()

    correct += (predicted_np == labels_np).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

