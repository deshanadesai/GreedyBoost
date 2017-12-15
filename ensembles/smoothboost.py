from math import e

class SmoothBoost(object):
	def __init__(self, learners, classes, total_points):
		self.classes = classes
		self.M = total_points
		self.learners = [learners(classes) for i in range(self.M)]

	def update(self, X, y):
		f = 0
		w = 0.5

		for learner in self.learners:
			learner.partial_fit(X, y, sample_weight = w)
			f += learner.predict(X)
			w = 1.0 / (1.0 + e**(y*f))

	def classify(self, X):
		label_weights = {}
		for i, learner in enumerate(self.learners):
			pred_y = learner.predict(X)
			if pred_y in label_weights.keys():
				label_weights[pred_y] += 1
			else:
				label_weights[pred_y] = 1
	        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))

