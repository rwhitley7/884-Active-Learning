# Introduction

Modern technology has provided an enormous amount of data that can be utilized for machine learning applications. However, for supervised learning tasks, this creates larger labeling and computational costs. For this reason, techniques such as Active Learning and Coresets have been heavily explored to remove data that provides the least information to the model. This allows large datasets to be summarized into smaller subsets that captures the same information, while maintaining similar performance levels. We explore different combinations of Coresets and Active learning techniques on the CIFAR10/100 datasets to provide quantitative analysis that can be leveraged for future supervised machine learning tasks.

# Coresets (Image/dataset summarization)

Coresets/Images summarization are techniques used to create a subset of images that are much smaller than the dataset that still capture similar information and can be used to reduce labeling and computational costs.

Some papers on Coresets/Image summarization:

Bayesian Batch Active Learning as Sparse Subset Approximation: https://arxiv.org/pdf/1908.02144.pdf


