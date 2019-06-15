import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models.resnet import model_urls

transform_test = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(p=0.2),
     transforms.RandomRotation(degrees=8),
     transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_batch_num = 100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_num, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_num, shuffle=True, num_workers=2)


def resnet18(pretrained = True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./'))
    return model


resnet_model = resnet18()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 100)
resnet_model.cuda()

# net = torch.load('net_L.pkl')

criterion = nn.CrossEntropyLoss()
LR = 0.001
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)


#####TRAINGING
for epoch in range(20):
    scheduler.step()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = resnet_model(inputs)
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
    #         outputs = resnet_model(images)
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
    outputs = resnet_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_cuda.size(0)
    predicted_np = predicted.cpu().numpy()
    labels_np = labels.numpy()

    correct += (predicted_np == labels_np).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# torch.save(resnet_model,'net_ljy.pkl')