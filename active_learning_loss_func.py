from random import random

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

from resnet import ResNet18
from torch.backends import cudnn


def downloadDataset():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
    }

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=data_transforms['train'])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset

def train_baseline(net, trainloader, device):


    # choose loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(1):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #with torch.set_grad_enabled(True):
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()
        #print(f"train_acc: ", train_acc)

    return net, criterion

def test(dataset, net, device, criterion):
    test_loss = 0
    correct = 0
    total = 0
    max_prob_tensor = torch.empty(1).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            max_prob, predicted = outputs.max(1)
            max_prob_tensor = torch.cat((max_prob_tensor, max_prob))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    return test_acc, max_prob_tensor.cpu()


if __name__ == '__main__':
    trainset, testset = downloadDataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #load test set
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #train seed data 10%
    train, val = torch.utils.data.random_split(trainset, [int(0.1 * len(trainset)), int(0.9 * len(trainset))])

    indices = list(range(len(trainset)))
    random.shuffle(indices)
    labeled_set = indices[:int(0.1 * len(trainset))]
    unlabeled_set = indices[int(0.1 * len(trainset)):]



    for i in range(2, 11):
        print(f"size of new validation set: ", len(val))

        trainloader = DataLoader(train, batch_size=128,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)

        valload = torch.utils.data.DataLoader(
                val, num_workers=2,
                batch_sampler=BatchSampler(SequentialSampler(val), batch_size=100, drop_last=False))
        net = ResNet18()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        net, criterion = train_baseline(net, trainloader, device)
        net.eval()
        test_acc, _ = test(testloader, net, device, criterion)

        print(f"for {i} iteration, test accuracy: ", test_acc)


        #active learning
        _, max_prob = test(valload, net, device, criterion)
        _, indices = torch.sort(max_prob) #sort in ascending order

        #choose the lowest confidence data samples for labelling
        low_conf_train = torch.utils.data.Subset(val, indices.numpy()[0:int(0.1*len(trainset))])
        train = torch.utils.data.ConcatDataset([train, low_conf_train])
        print(f"size of new train set: ", len(train))

        new_val = val[indices.numpy()[int(0.1*len(trainset)): ]]#torch.utils.data.Subset(val, indices.numpy()[int(0.1*len(trainset)): ])








