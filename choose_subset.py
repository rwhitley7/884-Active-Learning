import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import pickle

# load matrix with k nearest neighbors for each image
mat = np.load('topk-train-neighbors500.npy')

transform = transforms.Compose([transforms.ToTensor()])
# Download the 50,000 training images as 'train_dataset'
train_dataset = CIFAR10(root='data/',download=True,transform=transform)

# split data into different classes
class0,class1,class2,class3,class4,class5,class6,class7,class8,class9 = [],[],[],[],[],[],[],[],[],[]

for i in range(len(train_dataset)):
    if train_dataset[i][1] == 0:
        class0.append(i)
    if train_dataset[i][1] == 1:
        class1.append(i)
    if train_dataset[i][1] == 2:
        class2.append(i)
    if train_dataset[i][1] == 3:
        class3.append(i)
    if train_dataset[i][1] == 4:
        class4.append(i)
    if train_dataset[i][1] == 5:
        class5.append(i)
    if train_dataset[i][1] == 6:
        class6.append(i)
    if train_dataset[i][1] == 7:
        class7.append(i)
    if train_dataset[i][1] == 8:
        class8.append(i)
    if train_dataset[i][1] == 9:
        class9.append(i)

# generate subset for each class
c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = [],[],[],[],[],[],[],[],[],[]
c0_av,c1_av,c2_av,c3_av,c4_av,c5_av,c6_av,c7_av,c8_av,c9_av = [],[],[],[],[],[],[],[],[],[]

for row in class0:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c0_av:
        continue
    else:
        c0.append(idx1)
        c0_av.extend([idx2])

for row in class1:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c1_av:
        continue
    else:
        c1.append(idx1)
        c1_av.extend([idx2])

for row in class2:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c2_av:
        continue
    else:
        c2.append(idx1)
        c2_av.extend([idx2])

for row in class3:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c3_av:
        continue
    else:
        c3.append(idx1)
        c3_av.extend([idx2])

for row in class4:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c4_av:
        continue
    else:
        c4.append(idx1)
        c4_av.extend([idx2])

for row in class5:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c5_av:
        continue
    else:
        c5.append(idx1)
        c5_av.extend([idx2])

for row in class6:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c6_av:
        continue
    else:
        c6.append(idx1)
        c6_av.extend([idx2])

for row in class7:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c7_av:
        continue
    else:
        c7.append(idx1)
        c7_av.extend([idx2])

for row in class8:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c8_av:
        continue
    else:
        c8.append(idx1)
        c8_av.extend([idx2])

for row in class9:
    idx1,idx2,idx3,idx4,idx5 = mat[row][0],mat[row][1],mat[row][2],mat[row][3],mat[row][4]
    idx6,idx7,idx8,idx9,idx10,idx11 = mat[row][5],mat[row][6],mat[row][7],mat[row][8],mat[row][9],mat[row][10]
    if idx1 in c9_av:
        continue
    else:
        c9.append(idx1)
        c9_av.extend([idx2])

print("Length of each class = ", len(c0),len(c1),len(c2),len(c3),len(c4),len(c5),len(c6),len(c7),len(c8),len(c9))

total_subset = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
print("Length of total subset = ",len(total_subset))

# save indices to test
#with open("indices1n", "wb") as fp:
#   pickle.dump(total_subset, fp)

# choose random subset for active learning
rand_sub0 = random.sample(c0, 500)
rand_sub1 = random.sample(c1, 500)
rand_sub2 = random.sample(c2, 500)
rand_sub3 = random.sample(c3, 500)
rand_sub4 = random.sample(c4, 500)
rand_sub5 = random.sample(c5, 500)
rand_sub6 = random.sample(c6, 500)
rand_sub7 = random.sample(c7, 500)
rand_sub8 = random.sample(c8, 500)
rand_sub9 = random.sample(c9, 500)
total_random_subset = rand_sub0 + rand_sub1 + rand_sub2 + rand_sub3 + rand_sub4 + rand_sub5 + rand_sub6 + rand_sub7 + rand_sub8 + rand_sub9

with open("active_initial", "wb") as fp:
   pickle.dump(total_random_subset, fp)

print("Length of initial subset for active learning = ",len(total_random_subset))
#print(total_random_subset)
