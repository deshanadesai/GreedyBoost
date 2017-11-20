from random import shuffle
from collections import defaultdict
import numpy as np

def test(data, m, trials=1, shuffle=True):
	results = []
	for t in range(trials):
		if shuffle:
			shuffle(data)
		results.append(run_test(data, m)
		results = zip(*results)

    def avg(x):
        return sum(x) / len(x)
	return (map(avg, zip(*results[0])))
	
def run_test(data, m):
    classes = np.unique(np.array([y for (x, y) in data]))
    predictor = OzaBoostClassifier(classes=classes, M=m)
    correct = 0.0
    t = 0
    performance_booster = []
    for (features, label) in data:
        if predictor.classify(features) == label:
            correct += 1
        predictor.update(features, label)
        t += 1
        performance_booster.append(correct_booster / t)

	return performance_booster
