from collections import defaultdict
from math import log
import math
import numpy as np
from numpy.random import poisson, seed

from sklearn.tree import DecisionTreeClassifier
from sgdclassifier import sgd_classifier
from sklearn import tree

class DecisionTree(object):

	def __init__(self, classes):
		self.model = DecisionTreeClassifier(max_depth = 1)
		self.classes = classes
		self.X = None
		self.y = None

	def fit(self, x, y, classes):
		self.model.fit(x,y)

	def partial_fit(self, x, y):
		
		if self.X is None and self.y is None:
			self.X = x.reshape(1,-1)
			self.y = y.reshape(1,-1)
		else:
			#self.X = np.array(x.reshape(1, -1))
			self.X = np.vstack((self.X, x.reshape(1, -1)))
			#self.y = y.reshape(1, -1)
			self.y = np.vstack((self.y, y.reshape(1, -1)))

		self.model.fit(self.X,self.y)

	def predict(self, x):
		x = x.reshape(1,-1)
		try:
			y = self.model.predict(x)[0]
		except:
			return self.classes[0]
		return y

		
