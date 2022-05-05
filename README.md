# ECE 884 Project
## Introduction

Modern technology has provided an enormous amount of data that can be utilized for machine learning applications. However, for supervised learning tasks, this creates larger labeling and computational costs. For this reason, techniques such as Active Learning have been explored to reduce the amount of data needed to train a model. Since some data samples provide more information to the model than others, Active Learning looks to extract the most informative samples from the data pool iteratively. 

Many different metrics have been applied by researchers to determine which samples could provide most useful. In general, however, these techniques have been explored individually and in different environments, making comparisons challenging. In this project, we explore a few of the most popular Active Learning techniques using the same model and hyper-parameters to provide fair, quantitative analysis that can be leveraged for future supervised machine learning tasks. We also incorporate modern Deep Learning techniques, such as Contrastive Learning, in an attempt to further improve Active Learning.

We use the ResNet architecture as our backbone paired with the CIFAR datasets for training and classification. Information on the CIFAR datasets can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). We trained the model starting from a random 10% subset all the way up to the full dataset in 10% increments, with the accuracy values beeing shown below. These accuracies will be used as our baseline so that we can compare our techniques and determine whether our approach is beneficial or not. Our baseline is a modified version of [Pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

<p align="center">
  <img src="https://user-images.githubusercontent.com/47162612/164942566-ee440fe9-7bbd-49dd-87e7-6ee75ede7b4d.png" width="600" height="400">
</p>

## Active Learning

Active Learning can be summarized as follows:
1) Choose initial training subset, e.g. 10% of the training dataset
2) Use this subset to train the model
3) Test on the remainder of the training dataset, e.g. the other 90% of the training dataset
4) Use the results from testing to select next subset to add to the training subset, e.g choose the 10% that have the lowest confidence
5) Repeat 2-4 until desired accuracy is reached, e.g training subset is now 10% initially chosen + the 10% with the lowest confidence which will be tested on the remaining 80% of the training dataset to choose the next 10% of data to add

Often, the initial subset is chosen randomly. However, it is clear that different subsets will provide different results. For example, if the inital subset contains many similar images, the model will perform well when classifying other similar images but will struggle classifying a range of diverse images. As a result of this, part 1 of our projects looks at applying machine learning techniques to remove the most reduntant images to maximize the diversity of our initial subset.

## Part 1 - Using Contrastive Learning for Subset Initialization

[SCAN](https://arxiv.org/abs/2005.12320) groups images into semantically meanigful clusters without any labels. It is split in to two parts, with the first part being of interest to part 1 of our project. They use [SimCLR](https://arxiv.org/pdf/2002.05709.pdf), a self-supervised task that obtains semantically meaningful features via Contrastive Learning that are used to calculate each image's k-nearest neighbors. Leveraging the fact that each image in the dataset can be linked to other similar images, we attempt to remove as many of the closely related images so that we are left with the most diverse subset. Some examples of an image (far left image) and its 3 nearest neighbors can be seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47162612/163762551-b82f8ea8-75ee-4c14-8a6c-682e514b3cc6.png" width="400" height="400">
</p>

The 3 nearest neighbors of the first 2 images belong to the same class and share similar features. However, the 3rd image clusters its 3 nearest neighbors from a different class. It is easy to understand why this is done, as the majority of the images from the 'truck' class have their beds flat. Therefore, the features extracted from this 'truck' image are more closely related to the features of the 'airplane' images, such as the wings. These incorrect clusters from some images will not affect our work as we will only consider the nearest neighbors that belong to the same class.

As we are attempting to find a 10% subset that produces a higher accuracy than the random 10%, we removed different k values of neighbors to see their impacts. Each time we remove another k nearest neighbor, the subset gets smaller. Initially, with no neighbors removed (K = 0), the size of the CIFAR10 training dataset is 50k images. After removing each image's nearest neighbor (K = 1), our train subset was no reduced to ~35k images. We repeated this until we removed the 10 nearest neighbors (K = 10) for each images, reducing the size of the subset to ~14k images. From each subset, we chose a random 10% and tested on that to compare to the random 10% chosen from the full dataset. Our thoughs were that by removing images that were closely related to other images, we would be removing redundant data. This would give us a higher probability of choosing a random subset that is more diverse than if it was to be chosen from the full dataset. The graphs below show: (Left) The size of the subset with the different K neighbors removed, and (Right) the accuracy of the random 10% chosen from each subset.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/47162612/164944736-b97405e9-15f1-4814-882a-427bc43d6950.png">
  <img src="https://user-images.githubusercontent.com/47162612/164944738-bc673ce3-20c3-4f51-ac48-ac72edc41ac5.png">
</p>


## Part 2 - Different Active Learning Techniques

### Least Confidence:
After training on a subset of the training dataset, the model is tested on the remainder of the training dataset.
Choose the 10% of images that have the lowest confidence i.e. the ones the model struggles with the most.

### 1 vs 2:
1. After training on a subset of the training dataset, the model is tested on the remainder of the training dataset.
2. Calculate the difference in the confidence values for the top 2 classes.
3. Choose the 10% of images that have the lowest difference i.e. the ones that are close to decision boundaries.

### Learning Loss for Active Learning (LL4AL):
Loss Prediction Module (LPM) is connected to feature maps from target model.
LPM units consist of GAP, FC, ReLU.
During training iterations, LPM chooses which data to use, data is passed through the system, all weights are updated.

### KMeans:
1.	Extract features using transfer learning
2.	Cluster unlabelled data in k centres
3.	Start with 10% data closest to the cluster centres
4.	Choose 10% data farthest from cluster centres

### Pretext Tasks for Active Learning (PT4AL):
Initially, entire training set (50,000 images) is unlabeled and K = 1000.
ResNet-18 is used as the backbone network for the pretext task and the main task learner.



## Evaluation Results

### Active Learning : Method 1 - Least Confidence
<p align="center">
  <img src="https://user-images.githubusercontent.com/47718656/166968764-d46db83b-0a18-45c3-8870-ab4816794bba.jpg">
</p>

### Active Learning : Method 2 - 1 vs 2
<p align="center">
  <img src="https://user-images.githubusercontent.com/47718656/166969087-bf9404fe-78df-4171-b6a1-085957424316.png">
</p>

### Active Learning : Method 3 - LL4AL
<p align="center">
    <img src="https://user-images.githubusercontent.com/47718656/166969355-4c94c393-a42e-4e42-ad32-c55e04d60f17.jpg">
    <img src="https://user-images.githubusercontent.com/47718656/166969379-a1ddacc5-eb6e-4604-b856-09112a37776b.jpg">
</p>
  
### Active Learning : Method 4 - KMeans
<p align="center">
      <img src="https://user-images.githubusercontent.com/47718656/166969649-7a1ad987-8fea-4ea1-bfd2-387ebc84c851.jpg">
</p>

### Active Learning : Method 5 - PT4AL
<p align="center">
        <img src="https://user-images.githubusercontent.com/47718656/166970061-2545209a-9c1d-4bc5-ae2a-8ca0a71fef7c.jpg">
        <img src="https://user-images.githubusercontent.com/47718656/166970084-50a4d22c-1f18-4b6a-a77d-a502448d6b3a.jpg">  
</p>

### Comparison of AL Techniques
<p align="center">
          <img src="https://user-images.githubusercontent.com/47718656/166970424-643a79b4-a61a-49b7-8618-2a30260d203b.jpg">
</p>
PT4L achieves almost same accuracy as random 80% of the dataset using only 40%.  Our implementations achieve almost same accuracy as random 70% of the dataset using only 40%.  LL4AL and K-Means performed slightly worse than randomly choosing.

## Running Code
### Baseline results
The training and testing for baseline results can be found in "main.py." Run this code to select subset for training at random. Within this file, the amount of images to train and test can be specified as well as specific indices of images to use. Specific indices can be calculated using any Active Learning techniques.

### Least Confidence and 1v2
Choosing the indices with the lowest confidence or the smallest difference in the highest and second highest confidence can be found by runnning "python3 alearn.py." 1v2 is the current default, but uncommenting the line "s_inds = list(torch.topk(vals,10)[1].cpu().numpy())" will calculate the indices using the least confidence. After running the file, the new indices will be saved.

### K-Means based active learning
For feature extraction, run the following: <br>
`python feature_extractor.py`
This should save a file with name "features.npy". <br>

In order to perform k-means based active learning, run the following: <br>
`python k_means.py`
This uses the features generated in earlier step. Before running this, kindly change the path to kmeans_pytorch2 folder on line 12.

## References

* Extending Contrastive Learning to Unsupervised Coreset Selection: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/stamp/stamp.jsp?tp=&arnumber=9680708
* A Framework and Review: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/stamp/stamp.jsp?tp=&arnumber=9226466
* Understanding Contrastive Learning Requires Incorporating Inductive Biases: https://arxiv.org/pdf/2202.14037.pdf
* A Simple Framework for Contrastive Learning of Visual Representations: https://arxiv.org/abs/2002.05709 (code available at https://github.com/google-research/simclr)
* Unsupervised Learning of Visual Features by Contrasting Cluster Assignments: https://arxiv.org/abs/2006.09882 (code available at https://github.com/facebookresearch/swav)
* SCAN: Learning to Classify Images without Labels: https://arxiv.org/abs/2005.12320 (code available at https://github.com/wvangansbeke/Unsupervised-Classification)
* Local Aggregation for Unsupervised Learning of Visual Embeddings: https://arxiv.org/pdf/1903.12355.pdf
* Kmeans-pytorch: https://github.com/subhadarship/kmeans_pytorch
