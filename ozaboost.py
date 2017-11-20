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
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier():
	def __init__(self, classes, total_points):
		self.total_points = total_points
		self.learners = [DecisionTree(classes) for i in range(self.total_points)]
		self.correct = [0.0 for i in range(self.total_points)]
		self.incorrect = [0.0 for i in range(self.total_points)]
		self.error = [0.0 for i in range(self.total_points)]
		self.coeff = [0.0 for i in range(self.total_points)]


	def update(self, X, Y):
		weight = 1.0
		for i, learner in enumerate(learners):
			k = poisson(weight)
			for j in range(k):
				learner.partial_fit(X,Y)
			
			prediction = learner.predict(X)
			N = self.correct[i]+self.incorrect[i]

			if X==Y:
				self.correct[i] = self.correct[i]+weight
				weight = weight*(N/(2*self.correct[i]))
			else:
				incorrect[i] = incorrect[i]+weight
				weight = weight*(N/(2*incorrect[i]))

	def classify():	
		label_weights = {}	
		for i, learner in enumerate(learners):
			N = self.correct[i]+self.incorrect[i]
			self.error[i] = self.incorrect[m]/N
			self.coeff[i] = self.error[i]/(1.0-self.error[i])

		for i,learner in enumerate(learners):
			weight_learner = log(1/self.coeff[i])
			label = learner.predict(X)
			label_weights[label] += weight_learner

		return max(label_weights.iterkeys(), key = (lambda key: label_weights[key]))
			
class DecisionTree(object):

    def __init__(self, classes):
        self.model = DecisionTreeClassifier()
        self.X = None
        self.y = None

    def partial_fit(self, x, y):
        if self.X is None and self.y is None:
            self.X = x.toarray()
            self.y = y
        else:
            self.X = np.vstack((self.X, x.toarray()))
            self.y = np.hstack((self.y, y))

        self.model.fit(self.X, self.y)

    def predict(self, x):
	return self.model.predict(x.toarray())[0]
		
		
