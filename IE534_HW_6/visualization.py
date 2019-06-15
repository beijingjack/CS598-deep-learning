import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


########################### PART2 ###########################


###### plot a 10 by 10 grid of images scaled between 0 and 1
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
        x = self.conv2(x)
        x = self.ln4(x)
        x = F.leaky_relu(x)

        # conv5
        x = self.conv3(x)
        x = self.ln5(x)
        x = F.leaky_relu(x)

        #conv6
        x = self.conv3(x)
        x = self.ln6(x)
        x = F.leaky_relu(x)

        #conv7
        x = self.conv3(x)
        x = self.ln7(x)
        x = F.leaky_relu(x)

        # conv8
        x = self.conv2(x)
        x = self.ln8(x)
        x = F.leaky_relu(x)

        x = self.pool(x)

        output = x.view(-1, 196)
        output1 = self.fc1(output)
        output2 = self.fc10(output)

        return output1, output2

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

batch_size = 128

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)




model = torch.load('cifar10.model')
model.cuda()
model.eval()

# Grab a sample batch from the test dataset. Create an alternative label which is simply +1 to the true label.
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()


# save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

# Get the output from the fc10 layer and report the classification accuracy.
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

# Calculate the loss`` based on the alternative classes instead of the real classes.
## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)

# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

# evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

# save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)
