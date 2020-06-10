# ACTIVE-LEARNING-FRAMEWORK

#### Introduction

Active learning is a special branch of machine learning that falls under semi-supervised learning, wherein the machine continuously interacts with an information source (oracle). 

Most supervised machine learning models require large amounts of data to be trained with for good results. Getting labeled data is a difficult task. Considering the fact that data is abundantly available now-a-days, it is still very costly and time inefficient to label the data. Consider an image segmentation task, wherein we classify at the pixel level, it is very difficult for a human to provide the amount of labeled data required by a supervised model. What about we say we can achieve similar results with less amount of labeling. This is when Active Learning comes into picture.

Active Learning cleverly chooses the data points it wants to label and train its model on those points, leading to highest impact to training a supervised model. 

#### Implementation details

Implemented an Active Learning framework from scratch using basic Python libraries like NumPy and Pandas. The model is completely generalized and can be used for a variety of datasets.
