import numpy as np
from sklearn import linear_model

class sgd_classifier(object):
	def __init__(self, classes):
		self.classes = classes
		self.model = linear_model.SGDClassifier(warm_start = True)
		self.X = None
		self.Y = None

	def fit(self, X, Y, classes):
		self.model.partial_fit(X, Y, classes = self.classes)

	def partial_fit(self, X, Y):
		X = X.reshape(1,-1)
		Y = Y.reshape(1,-1)
		self.model.partial_fit(X, Y, classes = self.classes)
	
	def predict(self, X, classes = None):
		X = X.reshape(1,-1)
		return self.model.predict(X)[0]
		
