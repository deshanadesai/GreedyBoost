from random import shuffle
from collections import defaultdict
import numpy as np
import ozaboost_dynamic

class Test():
	def __init__(self, X, y, m):
		data = zip(X,y)
		self.classes = np.unique(np.array([y for (x,y) in data]))
		self.predictor = ozaboost_dynamic.OzaBoostClassifier(classes = self.classes, total_points = m)
		self.correct = 0.0
		self.t = 0
		self.baseline = ozaboost_dynamic.OzaBoostClassifier(classes = self.classes,total_points = 1)
	
	def test(self,X,y,X_val, y_val, m, trials=1, should_shuffle=True):
		results = []
		data = zip(X,y)
		'''errors = self.predictor.pretrain(X, y, X_val, y_val)
		baseline_error = self.baseline.pretrain(X,y,X_val, y_val)
		
		(baseline_error_x, baseline_error_y) = baseline_error[0]
		print "Baseline pre-training - Training Accuracy", baseline_error_x
		print "Baseline pre-training - Testing Accuracy", baseline_error_y
        
		for i, (training_error, testing_error) in enumerate(errors):
			print "Weak Learner",i," Training Accuracy: ", training_error
			print "Weak Learner",i," Testing Accuracy: ", testing_error'''
		for t in range(trials):
			if should_shuffle:
				shuffle(data)
			results.append(self.run_test(data))
		(booster, baseline) = results[-1]

		#print "Results: ",results	
		def avg(x):
			return sum(x)/len(x)
		return (avg(booster), avg(baseline))
	
	def run_test(self, data):
		performance_booster = []
		performance_baseline = []
		baseline_correct = 0.0
		prev_points = []
		clf_no = 0
		for (X,Y) in data:	
			if self.predictor.classify(X)[0] == Y:
				self.correct += 1
			weight = self.predictor.update(X,Y)

			(key, conf, label_weights) = self.predictor.classify(X)
			total = 0.0
			for (k,v) in label_weights.iteritems():
				total += v
			conf = float(conf)/total
			
			if (key!=Y and (conf<0.8 or conf==1.0))  or (key==Y and conf<0.5):
				clf_no +=1
				print ("Initializing new learner: ",clf_no)
				print (label_weights)
				print ("Initial confidence in incorrect classification",key," : ",conf, "& correct classification: ",Y)
				
				self.predictor.initialize_new_learner(X,Y,weight, prev_points)
			self.t += 1
			performance_booster.append(self.correct / self.t)
			if self.baseline.classify(X)[0] == Y:
				baseline_correct += 1
			self.baseline.update(X,Y)
			performance_baseline.append(baseline_correct / self.t)
			prev_points.append((X,Y))
		return performance_booster, performance_baseline

	def final_test(self,X,y,m):
		#print "Performing Test for",len(self.predictor)," weak learners.."
		correct_test = 0.0
		num_samples = 0.0
		data = zip(X,y)
		correct_baseline = 0.0
		for (X,y) in data:
			if self.predictor.classify(X)[0] == y:
				correct_test +=1
			num_samples +=1
			if self.baseline.classify(X)[0] == y:
				correct_baseline +=1
		avg = float(correct_test)/float(num_samples)
		baseline_avg = float(correct_baseline)/float(num_samples)
		return (avg,baseline_avg)
