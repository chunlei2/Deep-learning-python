import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


batch_size = 100
learning_rate = 0.001
num_epochs = 100
transform = transforms.Compose(
	[transforms.RandomCrop(32, padding = 4),
	 transforms.RandomHorizontalFlip(p = 0.5),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
									   download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										 shuffle=False, num_workers=2)



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(
						nn.Conv2d(3, 64, 4, stride = 1, padding = 2), #64*33*33
						nn.ReLU(),           
						nn.BatchNorm2d(64))        
		self.layer2 = nn.Sequential(
						nn.Conv2d(64, 64, 4, stride = 1, padding = 2), #64*34*34
						nn.ReLU(),
						nn.MaxPool2d(2, 2),
						nn.Dropout2d()) #64*17*17                        
		self.layer3 = nn.Sequential(
						nn.Conv2d(64, 64, 4, stride = 1, padding = 2), #64*18*18
						nn.ReLU(),           
						nn.BatchNorm2d(64))
		self.layer4 = nn.Sequential(
						nn.Conv2d(64, 64, 4, stride = 1, padding = 2), #64*19*19
						nn.ReLU(),            
						nn.MaxPool2d(2, 2),
						nn.Dropout2d()) #64*9*9
		self.layer5 = nn.Sequential(
						nn.Conv2d(64, 64, 4, stride = 1, padding = 2), #64*10*10
						nn.ReLU())
		self.layer6 = nn.Sequential(
						nn.Conv2d(64, 64, 3), #64*8*8
						nn.ReLU(),            
						nn.Dropout2d())
		self.layer7 = nn.Sequential(
						nn.Conv2d(64, 64, 3), #64*6*6
						nn.ReLU(),            
						nn.BatchNorm2d(64))
		self.layer8 = nn.Sequential(
						nn.Conv2d(64, 64, 3), #64*4*4
						nn.ReLU(),            
						nn.BatchNorm2d(64),
						nn.Dropout2d())
		self.fc1 = nn.Linear(64*4*4, 500)
		self.fc2 = nn.Linear(500, 500)
		self.fc3 = nn.Linear(500, 10)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = x.view(-1, 64*4*4)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x
# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = learning_rate * (0.1 ** (epoch // 100))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

net = Net()
net.cuda()
# dic = torch.load('two_300.ckpt')
# net.load_state_dict(dic)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):  # loop over the dataset multiple times
	# if epoch > 5:
	# 	learning_rate = 0.001
	# 	for param_group in optimizer.param_groups:
	# 		param_group['lr'] = learning_rate
	# elif epoch > 10:
	# 	learning_rate = 0.0001
	# 	for param_group in optimizer.param_groups:
	# 		param_group['lr'] = learning_rate
	# elif epoch > 15:
	# 	learning_rate = 0.00001
	# 	for param_group in optimizer.param_groups:
	# 		param_group['lr'] = learning_rate
	# adjust_learning_rate(optimizer, epoch)
	correct = 0
	total = 0
	net.train()
	for data in trainloader:
		# get the inputs
		inputs, labels = data
		inputs = Variable(inputs).cuda()
		labels = Variable(labels).cuda()

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
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
		images, labels = data
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()   
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.data.size(0)
		correct += (predicted == labels.data).sum()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))


torch.save(net.state_dict(), 'hw_03.ckpt')