import torch
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim

from torch.backends import cudnn
from torch.utils.data import SubsetRandomSampler, BatchSampler

from resnet import ResNet18

sys.path.insert(1, "C:\\Users\\protichi\\PycharmProjects\\884-Active-Learning\\kmeans_pytorch2")
from kmeans_pytorch import kmeans, kmeans_predict
import torchvision
from torchvision import transforms, models


def clustering(data, num_clusters):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    cluster_ids_x, cluster_centers, cluster_dist = kmeans(
        data, num_clusters=num_clusters, distance='euclidean', device=device
    )

    return cluster_ids_x, cluster_centers, cluster_dist


def train_dataload(indices):
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

    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=2,
        batch_sampler=BatchSampler(SubsetRandomSampler(indices), batch_size=500, drop_last=False))

    return trainloader


def test_dataloader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=800, shuffle=False, num_workers=2)

    return testloader


def train_baseline(trainloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    for epoch in range(200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()

        #print(f"train_acc: ", train_acc)
    test_acc = test(test_dataloader(), net, device, criterion)

    print(f"test_acc: ", test_acc)


def test(dataset, net, device, criterion):
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

    test_acc = 100. * correct / total
    return test_acc


if __name__ == '__main__':
    print(f"torch: ", torch.cuda.device_count())
    features_cifar10 = np.load("features.npy")
    features_cifar10_tensor = torch.from_numpy(features_cifar10)
    print("running for ascending order")
    cluster_ids_x, cluster_centers, cluster_dist = clustering(features_cifar10_tensor, 10)

    values, indices = torch.sort(cluster_dist, descending=True)
    #values, indices = torch.sort(cluster_dist)
    for i in range(1, 51):
        print(f"running for {i} iteration")

        indices_10 = indices[0:int(i * 0.02 * len(indices))]
        print(indices_10.size())

        trainData = train_dataload(indices_10)

        train_baseline(trainData)
