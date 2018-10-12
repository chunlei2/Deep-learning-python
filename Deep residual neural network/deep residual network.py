import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import model_zoo

#hw 04 part I
batch_size = 100
learning_rate = 0.001
num_epochs = 40
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding = 4),
     transforms.RandomHorizontalFlip(p = 0.5),
     transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2
)

def conv3x3(in_planes, out_planes, stride=1, downsample=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #convert to the same channel
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

blocks = [2, 4, 4, 2]
class ResNet(nn.Module):
    def __init__(self, block, blocks, num_classes = 100):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        self.layer1 = self._make_layer(block, 32, blocks[0])
        self.layer2 = self._make_layer(block, 64, blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[3], stride=2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256*2*2, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                                      nn.BatchNorm2d(planes)) #make same dimension of input and output
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(-1, 256*2*2)
        x = self.fc(x)
        return x
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = learning_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
net = ResNet(BasicBlock, blocks)
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    adjust_learning_rate(optimizer, epoch)
    net.train()
    for data in trainloader:
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Epoch %d:Accuracy of the network on the 50000 train images: %d %%' % (epoch,
       100 * correct / total))
    correct = 0
    total = 0
    net.eval()
    for data in testloader:
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
       100 * correct / total))
        
#hw04 part II
learning_rate = 0.0001
num_epochs = 5
model_urls = {
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}
def resnet18(pretrained = True) :
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained :
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
        return model

ResNet18 = resnet18()
ResNet18.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(ResNet18.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    ResNet18.train()
    for data in trainloader:
        images, labels = data
        m = nn.Upsample(scale_factor=7, mode='bilinear')
        images = m(images)
        images = images.data
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = ResNet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Epoch %d:Accuracy of the network on the 50000 train images: %d %%' % (epoch,
       100 * correct / total))
    correct = 0
    total = 0
    ResNet18.eval()
    for data in testloader:
        images, labels = data
        m = nn.Upsample(scale_factor=7, mode='bilinear')
        images = m(images)
        images = images.data
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = ResNet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
       100 * correct / total))                    


