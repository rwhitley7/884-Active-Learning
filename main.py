import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from utils import progress_bar
from torch.utils.data import random_split
from resnet import ResNet18

# Choose number of times to run training/testing 
num_runs = 10
total_train_accs = []
total_test_accs = []
val_size = 5000
train_size = 45000

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

	train_acc = 100.*correct/total
	return train_acc

def test(dataset):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(dataset):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	test_acc = 100.*correct/total
	return test_acc

for run in range(num_runs):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# prepare the data
	# add data augmentation to the training images for improved performance

	transform_train = transforms.Compose([
    		transforms.RandomCrop(32, padding=4),
    		transforms.RandomHorizontalFlip(),
    		transforms.ToTensor(),
    		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	transform_test = transforms.Compose([
    		transforms.ToTensor(),
    		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	trainset = torchvision.datasets.CIFAR10(
    		root='./data', train=True, download=True, transform=transform_train)

	train_split, val_split = random_split(trainset, [len(trainset)-val_size, val_size])
	train_split_use, train_split_disregard = random_split(train_split,[train_size,len(train_split)-train_size]) 

	trainloader = torch.utils.data.DataLoader(
    		train_split_use, batch_size=128, shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(
		val_split, batch_size=128, shuffle=False, num_workers=2)

	testset = torchvision.datasets.CIFAR10(
    		root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(
    		testset, batch_size=100, shuffle=False, num_workers=2)

	net = ResNet18()
	net = net.to(device)
	if device == 'cuda':
    		net = torch.nn.DataParallel(net)
    		cudnn.benchmark = True

	# choose loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.1,
		momentum=0.9, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

	for epoch in range(100):
		train_acc = train(epoch)
		eval_acc = test(valloader)
		scheduler.step()
	test_acc = test(testloader)
	total_train_accs.append(train_acc)
	total_test_accs.append(test_acc)
	print("Train accuracies = ",total_train_accs)
	print("Test accuracies = ",total_test_accs)

	print("Average train accuracies = ", sum(total_train_accs)/len(total_train_accs))
	print("Average test accuracies = ", sum(total_test_accs)/len(total_test_accs))
