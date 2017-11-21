from random import shuffle
from collections import defaultdict
import numpy as np
import ozaboost

class Test():
	def __init__(self, X, y, m):
		data = zip(X,y)
		self.classes = np.unique(np.array([y for (x,y) in data]))
		self.predictor = ozaboost.OzaBoostClassifier(classes = self.classes, total_points = m)
		self.correct = 0.0
		self.t = 0
	
	def test(self,X,y,m, trials=1, should_shuffle=True):
		results = []
		data = zip(X,y)
	
		for t in range(trials):
			if should_shuffle:
				shuffle(data)
			results.append(self.run_test(data))
	
		print "Results: ",results	
		def avg(x):
			return sum(x)/len(x)
		return avg(results[0])
	
	def run_test(self, data):
		performance_booster = []
		for (X,Y) in data:	
			if self.predictor.classify(X) == Y:
				self.correct += 1
			self.predictor.update(X,Y)
			self.t += 1
			performance_booster.append(self.correct / self.t)
		return performance_booster
	
