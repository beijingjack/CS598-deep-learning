import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.2),
     transforms.RandomRotation(degrees=8),
     transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_batch_num = 90
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_num, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_num, shuffle=True, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.dim_match1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.dim_match2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.dim_match3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.batch = nn.BatchNorm2d(32)
        self.drop = nn.Dropout2d(p=0.15)
        self.pool = nn.AvgPool2d(4, 1)  # !!!!! Own choice of size
        self.block_1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32)
        )
        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.block_3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128)
        )
        self.block_4 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = self.drop(x)
        residual = x
        # basic1 start
        x = self.conv1(x)
        x = self.block_1(x)
        x = x + residual
        # basic1 end
        residual = x
        # basic1 start
        x = self.conv1(x)
        x = self.block_1(x)
        x = x + residual
        # basic1 end
        residual = x
        # basic2 start
        x = self.conv2_1(x)
        x = self.block_2(x)
        residual = self.dim_match1(residual)
        x = x + residual
        # basic2 end
        residual = x
        # basic2 start
        x = self.conv2_2(x)
        x = self.block_2(x)
        x = x + residual
        # basic2 end
        residual = x
        # basic2 start
        x = self.conv2_2(x)
        x = self.block_2(x)
        x = x + residual
        # basic2 end
        residual = x
        # basic2 start
        x = self.conv2_2(x)
        x = self.block_2(x)
        x = x + residual
        # basic2 end
        residual = x
        # basic3 start
        x = self.conv3_1(x)
        x = self.block_3(x)
        residual = self.dim_match2(residual)
        x = x + residual
        # basic3 end
        residual = x
        # basic3 start
        x = self.conv3_2(x)
        x = self.block_3(x)
        x = x + residual
        # basic3 end
        residual = x
        # basic3 start
        x = self.conv3_2(x)
        x = self.block_3(x)
        x = x + residual
        # basic3 end
        residual = x
        # basic3 start
        x = self.conv3_2(x)
        x = self.block_3(x)
        x = x + residual
        # basic3 end
        residual = x
        # basic4 start
        x = self.conv4_1(x)
        x = self.block_4(x)
        residual = self.dim_match3(residual)
        x = x + residual
        # basic4 end
        residual = x
        # basic4 start
        x = self.conv4_2(x)
        x = self.block_4(x)
        x = x + residual
        # basic4 end
        x = self.pool(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        return x


net = Net()
net.cuda()

#net = torch.load('net_L.pkl')


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

#####TRAINGING
for epoch in range(20):
    scheduler.step()
    #train_accu = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if (epoch > 6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if (state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()

    print('Epoch ', epoch, ' Finished Training')

    # if epoch % 4 == 0:
    #     # print test accuracy after 4 epoch
    #     correct = 0
    #     total = 0
    #     # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         images, labels_cuda = Variable(images.cuda()), Variable(labels.cuda())
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels_cuda.size(0)
    #         predicted_np = predicted.cpu().numpy()
    #         labels_np = labels.numpy()
    #
    #         correct += (predicted_np == labels_np).sum()
    #
    #     print('Test Accuracy: %d %%' % (
    #             100 * correct / total))

#####TESTING
correct = 0
total = 0
# with torch.no_grad():
for data in testloader:
    images, labels = data
    images, labels_cuda = Variable(images.cuda()), Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_cuda.size(0)
    predicted_np = predicted.cpu().numpy()
    labels_np = labels.numpy()

    correct += (predicted_np == labels_np).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# torch.save(net,'net_ljy.pkl')