import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models.resnet import model_urls
import torchvision.datasets as datasets
import torch.utils.data
from PIL import Image
import os
import os.path
import random
import torch.nn.functional as F
import time
import math

# ######### May be used later
# train_dir = './tiny-imagenet-200/train'
# trainset = datasets.ImageFolder(train_dir, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_num, shuffle=True, num_workers=2)
#
base_path = './tiny-imagenet-200/train'
filenames_filename = './tiny-imagenet-200/wnids.txt'
###################### Build custom dataset
def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, transform=None,
                 loader=default_image_loader):
        self.base_path = base_path
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        class_index = int(index/500)
        file_index = index%500

        #anchor
        anchor_name = os.listdir(os.path.join(self.base_path,self.filenamelist[class_index],'images'))[file_index]
        #close
        r = list(range(0, file_index))+list(range(file_index + 1, 500))
        close_file_index = random.choice(r)
        close_name = os.listdir(os.path.join(self.base_path,self.filenamelist[class_index],'images'))[close_file_index]
        #far
        r = list(range(0, class_index)) + list(range(class_index + 1, 200))
        far_class_index = random.choice(r)
        r = range(0, 500)
        far_file_index = random.choice(r)
        far_name = os.listdir(os.path.join(self.base_path,self.filenamelist[far_class_index],'images'))[far_file_index]

        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[class_index],'images', anchor_name))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[class_index],'images', close_name))
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[far_class_index],'images', far_name))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        class_name = self.filenamelist[class_index]
        return img1, img2, img3, class_name

    def __len__(self):
        return 100000

###############Defining transform
transform_val = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

transform_train = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(p=0.2),
     transforms.RandomRotation(degrees=8),
     transforms.ToTensor()])

tripletset = TripletImageLoader(base_path,filenames_filename,transform=transform_train) #################### Transform!!!!
trainloader = torch.utils.data.DataLoader(dataset=tripletset, batch_size=64, shuffle=True, num_workers=4)
# embeddingloader = torch.utils.data.DataLoader(dataset=tripletset, batch_size=1, shuffle=False, num_workers=4)
# valset = ValImageLoader(val_base_path,transform=transform_val)
# valloader = torch.utils.data.DataLoader(dataset=ValImageLoader, batch_size=1, shuffle=False, num_workers=4)

# Defining networks
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, Q, P, N):
        embedded_Q = self.embeddingnet(Q)
        embedded_P = self.embeddingnet(P)
        embedded_N = self.embeddingnet(N)
        dist_P = F.pairwise_distance(embedded_Q, embedded_P, 2)
        dist_N = F.pairwise_distance(embedded_Q, embedded_N, 2)
        return dist_P, dist_N, embedded_Q, embedded_P, embedded_N

# Defining pretrained model, later can try 101
def resnet18(pretrained = True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./'))
    return model

############## Initialize network
resnet_model = resnet18()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 4096)
#resnet_model = torch.load('resnet_ljy.pkl')

tripnet = Tripletnet(resnet_model)
####### To GPU
tripnet.cuda()



criterion = torch.nn.MarginRankingLoss(margin=1)
optimizer = optim.SGD(resnet_model.parameters(), lr = 0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#####TRAINGING
#Switch to train mode
resnet_model.train()
tripnet.train()

for epoch in range(12):
    time1 = time.time()
    running_loss = 0.0
    scheduler.step()
    print ('Epoch ',epoch,'Start Training :)')
    for i, (data1,data2,data3,image_class) in enumerate(trainloader,0):
        # get the inputs
        # To GPU
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # Compute Output
        distP, distN, embedded_Q, embedded_P, embedded_N = tripnet(data1, data2, data3)

        # -1 means, distP should be smaller than distN
        target = torch.FloatTensor(distP.size()).fill_(-1)
        # To GPU
        target = target.cuda()
        target = Variable(target)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        triplet_loss = criterion(distP, distN, target)
        #embedding_loss = embedded_Q.norm(2) + embedded_P.norm(2) + embedded_N.norm(2)
        #ranking_loss = triplet_loss+0.0001*embedding_loss
        triplet_loss.backward()

        running_loss += triplet_loss.data[0]

    if epoch % 2 == 1:
        torch.save(resnet_model, 'resnet_test_ljy.pkl')

    time2 = time.time()
    print('Epoch ', epoch, ' Finish Training! ', 'Time = ',time2-time1)
    print ('Running loss = ',running_loss/100000)

