import numpy as np
from sampling import uncertainty_sampling, query_by_committee, stream_uncertainty_sampling, stream_query_by_committee
import random

class Learner:
	def __init__(self, committee, strategy, uncertainty = None, disagreement = None):
		''' initializing the Learner instance, based on the strategy and measure metric passed.
		'''
		self.committee = committee
		self.strategy = strategy
		self.uncertainty = uncertainty
		self.disagreement = disagreement

		if(len(committee) == 0):
			raise Exception("No model provided.")

		if(strategy == "uncertainty" and len(committee) > 1):
			warnings.warn("Multiple queries given in case of uncertainty sampling.")

	def train(self, orac):
		''' train the models in the committee with the available labeled data.
		'''
		numpy_X = np.array(orac.labeled_X)
		numpy_Y = np.array(orac.labeled_Y)
		
		for model in self.committee:
			model.fit(numpy_X, numpy_Y)

	def predict(self, X):
		''' predicts the class for the complete data of the Dataset instance.
		'''
		numpy_X = np.array(X)
		
		if self.strategy == "qbc":
			# if the strategy is query based committee, take the majority of predictions for each of the models.
			predictions = []
			for model in self.committee:
				pred = model.predict(numpy_X)
				predictions.append(pred)
			numpy_pred = np.transpose(np.array(predictions))
			final_predictions = []
			for instance in numpy_pred:
				cnt = np.bincount(instance)
				final_predictions.append(np.argmax(cnt))
			return np.array(final_predictions)

		else:
			return self.committee[0].predict(numpy_X)

	def test_score(self, data):
		''' runs the trained model on test data and get the score
		'''
		numpy_X = np.array(data.test_X)
		numpy_Y = np.array(data.test_Y)

		if(self.strategy == "qbc"):
			# if the strategy is query based committee, take the majority of predictions for each of the models.
			pred = self.predict(data.test_X)
			correct_pred = 0
			for i in range(pred.shape[0]):
				if(pred[i] == data.test_Y[i]):
					correct_pred = correct_pred + 1
			return correct_pred / pred.shape[0]
		else:
			return self.committee[0].score(numpy_X, numpy_Y)

	def query(self, data, orac, stream = -1, config = None):
		if(self.strategy == "uncertainty"):
			if(stream == -2):
				indices = np.array(random.sample(list(range(len(orac.pool_X))), 1))
			elif(stream != -1 and stream_uncertainty_sampling(self.uncertainty, self.committee[0], orac.pool_X[stream], config).any()):
				indices = stream
			elif(stream != -1):
				return 0
			elif(stream == -1):
				indices = uncertainty_sampling(data, orac, self.uncertainty, self.committee[0])
			orac.query(indices)
			return 1
		else:
			if(stream == -2):
				indices = np.array(random.sample(list(range(len(orac.pool_X))), 1))
			elif(stream != -1 and stream_query_by_committee(self.disagreement, self.committee, orac.pool_X[stream], config)):
				indices = stream
			elif(stream != -1):
				return 0
			elif(stream == -1):
				indices = query_by_committee(data, orac, self.disagreement, self.committee, config)
			orac.query(indices)
			return 1
