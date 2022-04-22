import copy
import time

import torchvision
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    premodel = models.resnet18(pretrained=True)
    print(premodel)

    for param in premodel.parameters():
        param.requires_grad = False

    num_ftrs = premodel.fc.in_features
    premodel.fc = nn.Linear(num_ftrs, 2)

    model_conv = premodel.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

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

    #choose random 10%
    #subset_train = torch.utils.data.random_split(trainset, [int(0.1*len(trainset)), int(0.9*len(trainset))])

    dataloaders = {'train': torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)}  # using entire train dataset


    dataset_sizes = {'train': len(trainset)}


    layer = premodel._modules.get('avgpool')
    premodel.eval()

    train_embeddings = numpy.zeros([len(trainset), 512])
    i = 0
    for j, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        for inp in inputs:
            my_embedding = torch.zeros(512)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data.reshape(o.data.size(1)))
            h = layer.register_forward_hook(copy_data)

            premodel(inp.unsqueeze(0))
            my_embedding = my_embedding.numpy()
            train_embeddings[i] = my_embedding
            #print(my_embedding)
            print(f"done for ", i)
            i += 1


    #numpy.save("features.npy", train_embeddings)




class feature_extractor:

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model




