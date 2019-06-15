import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# Constants
rotation_degree = 20
batchsize = 128
learning_rate = 0.001
epoch_num = 18
minibatch_size = 2000
image_channel_num = 3
conv_channel_num = 64
dropout_prob = 0.05


###### Make transformation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    transforms.ColorJitter(
        brightness=0.1 * torch.randn(1),
        contrast=0.1 * torch.randn(1),
        saturation=0.1 * torch.randn(1),
        hue=0.1 * torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
####### Download and construct CIFAR-10 datasets
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=8)

####### Discriminator network archetecture
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        self.ln1 = nn.LayerNorm((196, 32, 32))
        self.ln2 = nn.LayerNorm((196, 16, 16))
        self.ln3 = nn.LayerNorm((196, 16, 16))
        self.ln4 = nn.LayerNorm((196, 8, 8))
        self.ln5 = nn.LayerNorm((196, 8, 8))
        self.ln6 = nn.LayerNorm((196, 8, 8))
        self.ln7 = nn.LayerNorm((196, 8, 8))
        self.ln8 = nn.LayerNorm((196, 4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = F.leaky_relu(x)

        # conv4
        x = self.conv4(x)
        x = self.ln4(x)
        x = F.leaky_relu(x)

        # conv5
        x = self.conv5(x)
        x = self.ln5(x)
        x = F.leaky_relu(x)

        #conv6
        x = self.conv6(x)
        x = self.ln6(x)
        x = F.leaky_relu(x)

        #conv7
        x = self.conv7(x)
        x = self.ln7(x)
        x = F.leaky_relu(x)

        # conv8
        x = self.conv8(x)
        x = self.ln8(x)
        x = F.leaky_relu(x)

        x = self.pool(x)

        output = x.view(-1, 196)
        output1 = self.fc1(output)
        output2 = self.fc10(output)

        return output1, output2


####### Defining loss function and optimizer.
model = discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    running_loss = 0.0
    correct = 0
    total = 0

    if(epoch == 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 10.0
    if(epoch == 75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 100.0
    '''
    for i, data in enumerate(trainloader):

        inputs, labels = data
        if(labels.shape[0] < batchsize):
            continue

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        output = model(inputs)

        loss = criterion(output, labels)
        optimizer.zero_grad()
    '''
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        labels = Y_train_batch
        if (Y_train_batch.shape[0] < batchsize):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()

        if(epoch > 1):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()

        # train accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu().numpy() == labels.data.cpu().numpy()).sum().item()
    print('epoch: ', float(correct) / float(total))

print('Finished Training')
torch.save(model, 'cifar10.model')

####### Testing
# Should achieve between 87%-89% on the test set
correct = 0
total = 0
for data in testloader:
    optimizer.zero_grad()
    x_test, y_test = data
    x_test = Variable(x_test).cuda()
    _, outputs = model(x_test)
    _, predicted = torch.max(outputs.data, 1)
    total += y_test.size(0)
    correct += (predicted.cpu().numpy() == y_test.cpu().numpy()).sum().item()
print('Testing accuracy: %.5f', float(correct) / float(total))
print('Finished Testing')
