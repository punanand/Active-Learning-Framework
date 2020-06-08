import numpy as np
from math import floor, log
import random

def uncertainty_sampling(data, orac, measure, model):
	max_query_size = floor(0.1 * len(data.train_indices))
	
	numpy_X = np.array(orac.pool_X)

	predictions = model.predict_proba(numpy_X)
	predictions = -np.sort(-predictions, axis = 1)

	if measure == "lc":
		return np.argsort(predictions[:, 0])[:max_query_size]

	elif measure == "ms":
		return np.argsort(predictions[:, 0] - predictions[:, 1])[:max_query_size]

	elif measure == "random":
		return np.array(random.sample(list(range(len(orac.pool_X))), max_query_size))

	else:
		lst = []
		for i in range(predictions.shape[0]):
			sm = 0.0
			pred = predictions[i]
			for j in range(pred.shape[0]):
				tmp = pred[j]
				tmp = tmp * log(pred[j], 2)
				sm = sm + tmp 
			lst.append(sm)
		return np.argsort(np.array(lst))[:max_query_size]

	return None

def stream_uncertainty_sampling(measure, model, query_data, config):
	zr = np.zeros(query_data.shape[0])
	tmp = []
	tmp.append(query_data)
	tmp.append(zr)
	query_data = np.array(tmp)

	prediction = model.predict_proba(query_data)
	prediction = -np.sort(-prediction, axis = 1)

	if measure == "lc":
		return prediction[0, 0] < config.lc_threshold

	elif measure == "ms":
		return prediction[0, 0] - prediction[0, 1] < config.ms_threshold

	else:
		lst = []
		sm = 0.0
		pred = prediction[0]
		for j in range(pred.shape[0]):
			tmp = pred[j]
			tmp = tmp * log(pred[j], 2)
			sm = sm + tmp 
		lst.append(-1*sm / log(pred.shape[0], 2))
		return lst[0] > config.entropy_threshold

	return None

def query_by_committee(data, orac, disagreement, committee):
	max_query_size = floor(0.1 * len(data.train_indices))
	numpy_X = np.array(orac.pool_X)
	predictions = np.array([model.predict_proba(numpy_X) for model in committee])

	if disagreement == "kl":
		lst = []
		for instance in range(predictions.shape[1]):
			sum_over_models = 0
			for i in range(predictions.shape[0]):
				sum_over_classes = 0
				for j in range(predictions.shape[2]):
					sum_denominator = 0
					for k in range(predictions.shape[0]):
						sum_denominator = sum_denominator + predictions[k, instance, j]
					sum_denominator = sum_denominator / len(committee)

					tmp = predictions[i, instance, j]
					tmp = tmp / sum_denominator
					if(tmp != 0):
						tmp = log(tmp, 2)
						tmp = tmp * predictions[i, instance, j]
					sum_over_classes = sum_over_classes + tmp

				sum_over_models = sum_over_models + sum_over_classes
			sum_over_models = sum_over_models / len(committee)
			lst.append(-1*sum_over_models)

		return np.argsort(np.array(lst))[:max_query_size]

	else:
		lst = []
		prediction = np.array([model.predict(numpy_X) for model in committee])
		for instance in range(predictions.shape[1]):
			cnt = np.zeros(predictions.shape[2])
			for i in range(len(committee)):
				cnt[prediction[i, instance]] = cnt[prediction[i, instance]] + 1
			sm = 0
			for i in range(predictions.shape[2]):
				tmp = cnt[i] / len(committee)
				if(tmp != 0):
					tmp = tmp * log(tmp, 2)
				sm = sm + tmp
			lst.append(-1*sm)

		return np.argsort(np.array(lst))[:max_query_size]

def stream_query_by_committee(disagreement, committee, query_data, config):
	zr = np.zeros(query_data.shape[0])
	tmp = []
	tmp.append(query_data)
	tmp.append(zr)
	query_data = np.array(tmp)

	predictions = np.array([model.predict_proba(query_data) for model in committee])

	if disagreement == "kl":
		lst = []
		
		sum_over_models = 0
		for i in range(predictions.shape[0]):
			sum_over_classes = 0
			for j in range(predictions.shape[2]):
				sum_denominator = 0
				for k in range(predictions.shape[0]):
					sum_denominator = sum_denominator + predictions[k, 0, j]
				sum_denominator = sum_denominator / len(committee)

				tmp = predictions[i, 0, j]
				tmp = tmp / sum_denominator
				if(tmp != 0):
					tmp = log(tmp, 2)
					tmp = tmp * predictions[i, 0, j]
				sum_over_classes = sum_over_classes + tmp

			sum_over_models = sum_over_models + sum_over_classes
		sum_over_models = sum_over_models / len(committee)
		lst.append(sum_over_models)
		return lst[0] > config.kl_threshold

	else:
		lst = []
		prediction = np.array([model.predict(query_data) for model in committee])
		cnt = np.zeros(predictions.shape[2])
		for i in range(len(committee)):
			cnt[prediction[i, 0]] = cnt[prediction[i, 0]] + 1
		sm = 0
		for i in range(predictions.shape[2]):
			tmp = cnt[i] / len(committee)
			if(tmp != 0):
				tmp = tmp * log(tmp, 2)
			sm = sm + tmp
		lst.append(-1*sm)
		return lst[0] > config.ve_threshold



