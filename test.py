from random import shuffle
from collections import defaultdict
import numpy as np
import ozaboost

def test(X,y, m, trials=1, should_shuffle=True):
	results = []
	data = zip(X, y)
	for t in range(trials):
		if should_shuffle:
			shuffle(data)
		results.append(run_test(data,m))
		results = zip(*results)
		
	def avg(x):
		return sum(x)/len(x)
	return map(avg, zip(*results[0]))
	
def run_test(data, m):
	classes = np.unique(np.array([y for (x,y) in data]))
	predictor = ozaboost.OzaBoostClassifier(classes=classes, total_points=m)
	correct = 0.0
	t = 0
	performance_booster = []

	for (X,Y) in data:	
		if predictor.classify(X) == Y:
			correct += 1
		predictor.update(X,Y)
		t += 1
		performance_booster.append(correct / t)
	return performance_booster
	
