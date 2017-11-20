#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`Ozaboost`
==================

.. module:: Ozaboost
   :platform: Mac OS X
   :synopsis:

.. moduleauthor:: deshana.desai@nyu.edu

Created on 2017-11-19, 5:00

"""

from math import log
import numpy as np
from numpy.random import poisson, seed

from sklearn.tree import DecisionTreeClassifier


class OzaBoostClassifier():
	def __init__(self, classes, total_points):
		self.total_points = total_points
		self.classes = classes
		self.learners = [DecisionTree(classes) for i in range(self.total_points)]
		self.correct = [0.0 for i in range(self.total_points)]
		self.incorrect = [0.0 for i in range(self.total_points)]
		self.error = [0.0 for i in range(self.total_points)]
		self.coeff = [0.0 for i in range(self.total_points)]

	def update(self, X, Y):
		weight = 1.0
		for i, learner in enumerate(self.learners):
			print "Round: ",i
			k = poisson(weight)
			print "Poisson Dist: ",k
			if not k:
				continue

			for j in range(k):
				learner.partial_fit(X,Y)
			
			prediction = learner.predict(X)
			
			print "Initial Weight: ", weight

			if prediction == Y:
				self.correct[i] = self.correct[i]+weight
				N = float(self.correct[i]+self.incorrect[i])
				temp = (N/(2.0*float(self.correct[i])))
				weight *= temp
				print "Weight of example has decreased to: ",weight
			else:
				self.incorrect[i] = self.incorrect[i]+weight
				N = float(self.correct[i]+self.incorrect[i])
				weight = weight*(N/(2*self.incorrect[i]))
				print "Weight of example has increased to: ",weight

	def classify(self,X):	
		label_weights = {}	
		for i, learner in enumerate(self.learners):
			N = self.correct[i]+self.incorrect[i]+ 1e-16
			self.error[i] = (self.incorrect[i]+ 1e-16)/N
			self.coeff[i] = self.error[i]+ 1e-16/(1.0-self.error[i]+ 1e-16)

		for i,learner in enumerate(self.learners):
			weight_learner = log(1/self.coeff[i])
			label = learner.predict(X, self.classes)
			if label in label_weights.keys():
				label_weights[label] += weight_learner
			else:
				label_weights[label] = weight_learner

		return max(label_weights.iterkeys(), key = (lambda key: label_weights[key]))
			
class DecisionTree(object):

	def __init__(self, classes):
		self.model = DecisionTreeClassifier(max_depth = 1, random_state = 1)
		self.X = None
		self.y = None

	def partial_fit(self, x, y):
		
		if self.X is None and self.y is None:
			self.X = x.reshape(1,-1)
			self.y = y.reshape(1,-1)
		else:
			x = np.array(x.reshape(1, -1))
			self.X = np.vstack((self.X, x))
			y = y.reshape(1, -1)
			self.y = np.vstack((self.y, y))

		#print self.X
		#print self.y


		self.model.fit(self.X,self.y)

	def predict(self, x, classes = None):
		x = x.reshape(1,-1)
		try:
			y = self.model.predict(x)[0]
		except:
			y = classes[0]
		return y

		
