from collections import defaultdict
from math import log
import math
import numpy as np
from numpy.random import poisson, seed

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

class DecisionTree(object):

	def __init__(self, classes):
		self.model = DecisionTreeClassifier(max_depth = 1)
		self.classes = classes
		self.X = None
		self.y = None
		self.sample_weight = []

	'''
	def fit(self, x, y, sample_weight):
		self.model.fit(x,y, sample_weight = self.sample_weight/sum(self.sample_weight) )
	'''

	def partial_fit(self, X, y, sample_weight =1.0):
		
		if self.X is None and self.y is None:
			self.X = X.reshape(1,-1)
			self.sample_weight.append(sample_weight)
			self.y = y.reshape(1,-1)
		else:
			#self.X = np.array(x.reshape(1, -1))
			self.X = np.vstack((self.X, X.reshape(1, -1)))
			#self.y = y.reshape(1, -1)
			self.sample_weight.append(sample_weight)
			self.y = np.vstack((self.y, y.reshape(1, -1)))

		wts = [x/sum(self.sample_weight) for x in self.sample_weight]
		self.model.fit(self.X,self.y, sample_weight= wts)

	def predict(self, x):
		x = x.reshape(1,-1)
		try:
			y = self.model.predict(x)[0]
		except:
			return self.classes[0]
		return y

		
