import numpy as np
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.utils.data
from PIL import Image
import os
import os.path
import random
from sklearn.neighbors import NearestNeighbors
import torch
import time



#Load the trained resnet model
# resnet_model = torch.load('resnet_ljy.pkl')
# resnet_model.cuda()
# resnet_model.eval()
print("New Version")
#### For training images
base_path = './tiny-imagenet-200/train'
filenames_filename = './tiny-imagenet-200/wnids.txt'


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
        print (anchor_name, class_name)
        return img1, img2, img3, class_name

    def __len__(self):
        return 100000


#### For validation images
val_base_path = './tiny-imagenet-200/val/images'
val_txt = './tiny-imagenet-200/val/val_annotations.txt'
class ValImageLoader(torch.utils.data.Dataset):
    def __init__(self, val_base_path, val_txt_path, transform=None,
                 loader=default_image_loader):
        self.base_path = val_base_path
        self.transform = transform
        self.loader = loader
        self.val_path = val_txt_path

    def __getitem__(self, index):
        #######Use index in the txt file to get the string
        # Get img name and class label in string
        for i,line in enumerate(open(self.val_path)):
            if i == index:
                name = line.split()[0]
                label = line.split()[1]

        img = self.loader(os.path.join(self.base_path, name))
        if self.transform is not None:
            img = self.transform(img)

        return img,label

    def __len__(self):
        return 10000

transform_train = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(p=0.2),
     transforms.RandomRotation(degrees=8),
     transforms.ToTensor()])
transform_val = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

tripletset = TripletImageLoader(base_path,filenames_filename,transform=transform_train) #################### Transform!!!!
embeddingloader = torch.utils.data.DataLoader(dataset=tripletset, batch_size=1, shuffle=True, num_workers=2)
valset = ValImageLoader(val_base_path,val_txt,transform=transform_val)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=True, num_workers=2)

for i, (data1,data2,data3,image_class) in enumerate(embeddingloader,0):
    if i>=5:
        break
    print (i)
