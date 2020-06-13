# Active Learning Framework

### About Active Learning

Active learning is a special branch of machine learning that falls under semi-supervised learning, wherein the machine continuously interacts with an information source (oracle). 

Most supervised machine learning models require large amounts of data to be trained with for good results. Getting labeled data is a difficult task. Considering the fact that data is abundantly available now-a-days, it is still very costly and time inefficient to label the data. Consider an image segmentation task, wherein we classify at the pixel level, it is very difficult for a human to provide the amount of labeled data required by a supervised model. What about we say we can achieve similar results with less amount of labeling. This is when Active Learning comes into picture.

Active Learning cleverly chooses the data points it wants to label and train its model on those points, leading to highest impact to training a supervised model. 

### Implementation

[Active-Learning-Framework](https://punanand.github.io/Active-Learning/ "Active Learning Framework") has been implemented using basic Python libraries. The framework is completely generalized and can be used for a variety of datasets.
This repository contains notebooks that demonstrate the following Active Learning Strategies:

1. Pool Based Active Learning
    1. Uncertainty Sampling ([PoolBased_UncertaintySampling .ipynb](https://github.com/punanand/Active-Learning/blob/master/PoolBased_UncertaintySampling%20.ipynb))
        1. Least Confident Measure
        2. Entropy Measure
        3. Most Confused Measure
    2. Query-by-Committee Sampling ([PoolBased_QBC.ipynb](https://github.com/punanand/Active-Learning/blob/master/PoolBased_QBC.ipynb))
        1. Vote Entropy Based Sampling
        2. KL Divergence Based Sampling
2. Stream Based Active Learning 
    1. Uncertainty Sampling ([StreamBased_UncertaintySampling.ipynb](https://github.com/punanand/Active-Learning/blob/master/StreamBased_UncertaintySampling.ipynb))
        1. Least Confident Measure
        2. Entropy Measure
        3. Most Confused Measure
    2. Query-by-Committee Sampling  ([StreamBased_QBC.ipynb](https://github.com/punanand/Active-Learning/blob/master/StreamBased_QBC.ipynb))
        1. Vote Entropy Based Sampling
        2. KL Divergence Based Sampling
3. Cluster based sampling ([Cluster-based-sampling.ipynb](https://github.com/punanand/Active-Learning/blob/master/Cluster-based-sampling.ipynb))       

Click [here](https://punanand.github.io/Active-Learning/ "Active Learning Framework") for a more detailed analysis and implementation specific details about the framework.
## How to use the framework
The framework has been tested in Python 3.7
### Requirements

* [NumPy](https://numpy.org/ "NumPy")
* [Pandas](https://pandas.pydata.org/ "pandas"). 
* [scikit-learn](https://scikit-learn.org/stable/ "scikit-learn")

To install the above requirements, execute the following in a terminal:

`$ pip install numpy pandas scikit-learn matplotlib`
