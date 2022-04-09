# Introduction

Modern technology has provided an enormous amount of data that can be utilized for machine learning applications. However, for supervised learning tasks, this creates larger labeling and computational costs. For this reason, techniques such as Active Learning and Coresets have been heavily explored to remove data that provides the least information to the model. This allows large datasets to be summarized into smaller subsets that captures the same information, while maintaining similar performance levels. We explore different combinations of Coresets and Active learning techniques on the CIFAR10/100 datasets to provide quantitative analysis that can be leveraged for future supervised machine learning tasks.

# Image Summarization / Coresets / Contrastive Learning

Coresets/Images summarization are techniques used to create a subset of images that are much smaller than the dataset that still capture similar information and can be used to reduce labeling and computational costs.

Some papers on Image Summarization:
* Image Corpus Representative Summarization: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/document/8919310 (code available at https://github.com/Anurag14/ImageCorpusSummarization however it is not explained at all)
* Image Summarization Using Unsupervised Learning: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/document/9441682
* Less is more: https://arxiv.org/abs/2104.12835

Some papers on Coresets:
* Extending Contrastive Learning to Unsupervised Coreset Selection: https://ieeexplore-ieee-org.proxy1.cl.msu.edu/stamp/stamp.jsp?tp=&arnumber=9680708

Some papers on Contrastive Learning:


# Active Learning

Some papers on active learning:
* Bayesian Batch Active Learning as Sparse Subset Approximation: https://arxiv.org/pdf/1908.02144.pdf


