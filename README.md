# Introduction

Modern technology has provided an enormous amount of data that can be utilized for machine learning applications. However, for supervised learning tasks, this creates larger labeling and computational costs. For this reason, techniques such as Active Learning have been explored to reduce the amount of data needed to train a model. Since some data samples provide more information to the model than others, Active Learning looks to extract the most informative samples from the data pool iteratively. 

Many different metrics have been applied by researchers to determine which samples could provide most useful. In general, however, these techniques have been explored individually and in different environments, making comparisons challenging. In this project, we explore a few of the most popular Active Learning techniques using the same model and hyper-parameters to provide fair quantitative analysis that can be leveraged for future supervised machine learning tasks. We also incorporate modern Deep Learning techniques, such as Contrastive Learning, in an attempt to further improve Active Learning.

We use the ResNet architecture as our backbone paired with the CIFAR datasets for training and classification. Information on the CIFAR datasets can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

# Active Learning

Active Learning can be summarized as follows:
1) Choose initial training subset, e.g. 10% of the training dataset
2) Use this subset to train the model
3) Test on the remainder of the training dataset, e.g. the other 90% of the training dataset
4) Use the results from testing to chose next subset to add to the training subset, e.g choose the 10% that have the lowest confidence
5) Repeat 2-4 until desired accuracy is reached, e.g training subset is now 10% initially chosen + the 10% with the lowest confidence which will be tested on the remaining 80% of the training dataset to choose the next 10% of data to add

Often, the initial subset is chosen randomly. However, it is clear that different subsets will provide different results. For example, if the inital subset contains many similar images, the model will perform well when classifying other similar images but will struggle classifying a range of diverse images. As a result of this, part 1 of our projects looks at applying machine learning techniques to remove the most reduntant images to maximize the diversity of our initial subset.

# Part 1 - Using Contrastive Learning for Subset Initialization

[SCAN](https://arxiv.org/abs/2005.12320) groups images into semantically meanigful clusters without any labels. It is split in to two parts, with the first part being of interest to part one of our project. They use [SimCLR](https://arxiv.org/pdf/2002.05709.pdf), a self-supervised task that obtains semantically meaningful features via Contrastive Learning that are used for calculating each image's k-nearest neighbors. Leveraging the fact that each image in the dataset can be linked to other similar images, we attempt to remove as many of the closely related images so that we are left with the most diverse subset. Some examples of the images can be seen below.



# Part 2 - Different Active Learning Techniques

Active Learning ...


# Relevant Papers and Github Repos

Coresets/Images Summarization/Contrastive Learning are techniques used to create a subset of images that are much smaller than the dataset that still capture similar information and can be used to reduce labeling and computational costs.

Some papers on Image Summarization:
* Image Corpus Representative Summarization: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/document/8919310 (code available at https://github.com/Anurag14/ImageCorpusSummarization however it is not explained at all)
* Image Summarization Using Unsupervised Learning: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/document/9441682
* Less is more: https://arxiv.org/abs/2104.12835

Some papers on Coresets:
* Extending Contrastive Learning to Unsupervised Coreset Selection: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/stamp/stamp.jsp?tp=&arnumber=9680708

Some papers on Contrastive Learning:
* A Framework and Review: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/stamp/stamp.jsp?tp=&arnumber=9226466
* Understanding Contrastive Learning Requires Incorporating Inductive Biases: https://arxiv.org/pdf/2202.14037.pdf
* A Simple Framework for Contrastive Learning of Visual Representations: https://arxiv.org/abs/2002.05709 (code available at https://github.com/google-research/simclr)
* Unsupervised Learning of Visual Features by Contrasting Cluster Assignments: https://arxiv.org/abs/2006.09882 (code available at https://github.com/facebookresearch/swav)

Some papers on Image Clustering:
* SCAN: Learning to Classify Images without Labels: https://arxiv.org/abs/2005.12320 (code available at https://github.com/wvangansbeke/Unsupervised-Classification)
* Local Aggregation for Unsupervised Learning of Visual Embeddings: https://arxiv.org/pdf/1903.12355.pdf

Some papers on active learning:
* Bayesian Batch Active Learning as Sparse Subset Approximation: https://arxiv.org/pdf/1908.02144.pdf


