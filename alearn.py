import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import models
import pickle
from resnet import ResNet18, ResNet50
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = CIFAR10(root='data/',download=True,transform=transform_train)
testset = CIFAR10(root='data/',download=False,transform=transform_test)

inds = pickle.load( open( "dif_active_80", "rb" ) )
print("Training set = ",len(inds))

inds_other = list(range(len(trainset)))

for i in inds:
	inds_other.remove(i)

print("Testing set = ",len(inds_other))

al_trainset = torch.utils.data.Subset(trainset, inds)
al_testset = torch.utils.data.Subset(trainset, inds_other)

batch_size = 20
al_trainloader = DataLoader(al_trainset, batch_size, shuffle=True)
al_testloader = DataLoader(al_testset, batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size, shuffle=False)

net = ResNet18()
net = net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True
net.load_state_dict(torch.load('dif_active_80_percent.pth'))
if torch.cuda.is_available():
    net.cuda()
criterion = nn.CrossEntropyLoss()

net.eval()
test_loss = 0
correct = 0
total = 0
inds_append = []
k = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(al_testloader):
            #print("Batch index = ",batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.shape)
            outputs = net(inputs)
            outputs_cpu = outputs.cpu().numpy()
            # using 1 vs 2
            top2 = torch.topk(outputs,2,axis=1)[1].cpu().numpy()
            #print(outputs)
            #print(top2)
            difs = []
            for i in range(batch_size):
                row = outputs_cpu[i]
                ind1 = top2[i][0]
                ind2 = top2[i][1]
                dif = np.abs(row[ind1] - row[ind2])
                difs.append(dif)
            sorted = np.argsort(np.array(difs))
            s_inds = list(sorted[:10])
            #print(difs)
            #print(sorted)
            #print(outputs.shape)
            #print(outputs)
            loss = criterion(outputs, targets)
            #print(targets.shape)
            #print(targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(_.shape)
            #print(predicted)
            #print("Top 3 largest = ",torch.topk(_,3))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            vals = -1*_
            #print(vals)
            #s_inds = list(torch.topk(vals,10)[1].cpu().numpy())
            small_inds = [i+(batch_idx*batch_size) for i in s_inds]
            #print("Bottom 10 smallest  = ",small_inds)
            inds_append.extend(small_inds)

print("Inds append length = ", len(inds_append))
#print(inds_append)

test_acc = 100.*correct/total
print("Test Acc = ",test_acc)

inds_update = []
for i in inds_append:
    inds_update.append(inds_other[i])

inds_active = inds + inds_update

with open("dif_active_90", "wb") as fp:
     pickle.dump(inds_active, fp)
