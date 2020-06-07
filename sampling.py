import numpy as np
from math import floor, log

def uncertainty_sampling(data, orac, measure, model):
	max_query_size = floor(0.1 * len(data.x_raw))
	
	numpy_X = np.array(orac.pool_X)

	predictions = model.predict_proba(numpy_X)
	np.sort(predictions, axis = 1)

	if measure == "lc":
		return np.argsort(predictions[:, 0])[:max_query_size]

	elif measure == "ms":
		return np.argsort(predictions[:, 0] - predictions[:, 1])[:max_query_size]

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