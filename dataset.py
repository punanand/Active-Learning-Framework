from sklearn.datasets import load_digits
from math import floor
import random

class Config:
	def __init__(self, train_proportion, label_proportion):
		# proportion of trained data
		self.train_proportion = train_proportion
		# initial proportion of labeled data
		self.label_proportion = label_proportion
		# Maximum number of records in each query
		self.max_query_size = 5
		# Threshold for least count uncertainty measure
		self.lc_threshold = 0.5
		# Threshold for marginal sampling uncertainty measure
		self.ms_threshold = 0.2
		# Threshold for entropy uncertainty measure
		self.entropy_threshold = 0.5
		# threshold for kl divergence stream based sampling
		self.kl_threshold = 0.4
		# threshold for vote entropy stream based sampling
		self.ve_threshold = 0.6
		# proportion of data to take for cluster based sampling
		self.cluster_prop = 0.6
		# number of clusters to make during cluster based sampling
		self.num_clusters = 10

class Dataset:
	def __init__(self, config):
		# loading the data
		data = load_digits()
		self.x_raw = data['data']
		self.y_raw = data['target']
		
		# partition into test and train
		lst = list(range(self.x_raw.shape[0]))
		self.train_indices = random.sample(lst, floor(config.train_proportion * len(lst) / 100))
		test_lst = []
		for i in range(len(lst)):
			if i not in self.train_indices:
				test_lst.append(i)
		self.test_indices = test_lst

		self.test_X = [self.x_raw[i] for i in self.test_indices]
		self.test_Y = [self.y_raw[i] for i in self.test_indices]