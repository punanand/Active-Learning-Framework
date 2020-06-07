from sklearn.datasets import load_digits
from math import floor
import random

class Config:
	def __init__(self, train_proportion, label_proportion):
		self.train_proportion = train_proportion
		self.label_proportion = label_proportion

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