from collections import defaultdict
import numpy as np

class OSBoost(object):

	def __init__(self, learners, classes, total_points = 10, gamma = 0.1):
		self.M = total_points
		self.learners = [learners(classes) for i in range(self.M)]
		self.alpha = np.ones(self.M)/self.M
		self.gamma = gamma
		self.theta = self.gamma/(2 + self.gamma)


	def update(self, X, y):
		z_t = 0.0
		w = 1.0
		for learner in self.learners:
			z_t += learner.predict(X)*y - self.theta
			learner.partial_fit(X, y, sample_weight = w)
			if z_t <=0:
				w = 1.0
			else:
				w = (1.0 - self.gamma)** (z_t/2.0)

	def raw_predict(self, X):
		return sum(learner.predict(X) for learner in self.learners)

	def classify(self, X):
		label_weights = {}
		for i in range(self.M):
			pred_y = self.learners[i].predict(X)
			if pred_y in label_weights.iterkeys():
				label_weights[pred_y] += self.alpha[i]
			else:
				label_weights[pred_y] = self.alpha[i]

		key = max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
		return key
