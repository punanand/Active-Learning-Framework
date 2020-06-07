from math import floor
import random
import numpy as np

class Oracle:
	def __init__(self, data, config):
		# making a labeled - unlabeled partition.
		self.X = [data.x_raw[i] for i in data.train_indices]
		self.Y = [data.y_raw[i] for i in data.train_indices]
		labeled_indices = random.sample(list(range(len(self.X))), floor(config.label_proportion * len(self.X) / 100))
		self.labeled_X, self.labeled_Y, self.pool_X, self.pool_Y = [], [], [], []
		for i in range(len(self.X)):
			if i in labeled_indices:
				self.labeled_X.append(self.X[i])
				self.labeled_Y.append(self.Y[i])
			else:
				self.pool_X.append(self.X[i])
				self.pool_Y.append(self.Y[i])

	def query(self, indices_arr):
		indices = indices_arr.tolist()
		# returning the labels for the asked indices and add them to labeled data.
		for idx in indices:
			self.labeled_X.append(self.pool_X[idx])
			self.labeled_Y.append(self.pool_Y[idx])
		self.pool_X = [i for j, i in enumerate(self.pool_X) if j not in indices]
		self.pool_Y = [i for j, i in enumerate(self.pool_Y) if j not in indices]
		