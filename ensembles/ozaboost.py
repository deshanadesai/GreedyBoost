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
from random import shuffle
from collections import defaultdict
from math import log
import math
import numpy as np
from numpy.random import poisson, seed

class OzaBoostClassifier():
	def __init__(self, learners, classes, total_points):
		self.total_points = total_points
		self.classes = classes
		self.learners = [learners(classes) for i in range(self.total_points)]
		#self.learners = [DecisionTree(classes) for i in range(self.total_points)]
		#self.learners = [perceptron.Perceptron(classes) for i in range(self.total_points)]
		#self.learners = [sgd_classifier(classes) for i in range(self.total_points)]
		self.correct = [0.0 for i in range(self.total_points)]
		self.incorrect = [0.0 for i in range(self.total_points)]
		self.error = [0.0 for i in range(self.total_points)]
		self.coeff = [0.0 for i in range(self.total_points)]

        def get_error_rate(self, predictions, Y):
			temp = zip(predictions, Y)
			correct = 0.0
			for (p,y) in temp:
				if p==y:
					correct+=1
			return float(correct)/float(len(Y))

	def pretrain(self, train_X, train_Y, X_val, y_val):
		errors = []
		import random
		data = zip(train_X, train_Y)
		for i, learner in enumerate(self.learners):
			datapoints_x = []
			datapoints_y = []
			while True:
				index_point = random.randint(0,train_X.shape[0]-1)
				datapoints_x.append(train_X[index_point])
				datapoints_y.append(train_Y[index_point])
				learner.fit(datapoints_x, datapoints_y, self.classes)
				test_error = self.get_error_rate(learner.model.predict(X_val), y_val)
				if test_error>0.5:
					break
				#train_X, train_Y, test_X, test_Y = train_test_split(data, test_size = 0.2)
			#learner.model.fit(datapoint_x, train_Y)
			#learner.model.fit(datapoints_x,datapoints_y)
			#test_error = learner.model.predict(X_val)
			training_error = learner.model.predict(train_X)
			errors.append((self.get_error_rate(training_error, train_Y),test_error))
		return errors

	def update(self, X, Y):
		weight = 1.0
		#shuffle(self.learners)
		for i, learner in enumerate(self.learners):
			#print "Round: ",i
			k = poisson(weight)
			#print "Poisson Dist: ",k
			if k<=0:
				continue

			for j in range(k):
				learner.partial_fit(X,Y)
			
			prediction = learner.predict(X)
			
			#tree.export_graphviz(learner.model, out_file = "learner_"+str(i)+".dot")
			#print "Initial Weight: ", weight

			if prediction == Y:
				self.correct[i] = self.correct[i]+weight
				N = float(self.correct[i]+self.incorrect[i])
				temp = (N/(2.0*float(self.correct[i])))
				
				weight *= temp
				#print "Weight of example has decreased to: ",weight
			else:
				self.incorrect[i] = self.incorrect[i]+weight
				N = float(self.correct[i]+self.incorrect[i])
				weight = weight*(N/(2*self.incorrect[i]))
				#print "Weight of example has increased to: ",weight

	def classify(self,X):	
		label_weights = {}
		
		for i, learner in enumerate(self.learners):
			N = self.correct[i]+self.incorrect[i]+ 1e-16
			self.error[i] = (self.incorrect[i]+ 1e-16)/N
			self.coeff[i] = (self.error[i]+ 1e-16)/(1.0-self.error[i]+ 1e-16)

		for i,learner in enumerate(self.learners):
			weight_learner = log(1/self.coeff[i])
			label = learner.predict(X)
			if label in label_weights.keys():
				label_weights[label] += weight_learner
			else:
				label_weights[label] = weight_learner
		#print("learner ",i,": ",format(weight_learner,"0.2f"))

		return max(label_weights.iterkeys(), key = (lambda key: label_weights[key]))
			
