from random import shuffle
from collections import defaultdict
import numpy as np
import ozaboost

class Test():
	def __init__(self, weak_learner, X, y, m):
		data = zip(X,y)
		self.classes = np.unique(np.array([y for (x,y) in data]))
		self.predictor = ozaboost.OzaBoostClassifier(learners = weak_learner, classes = self.classes, total_points = m)
		self.correct = 0.0
		self.t = 0
		self.baseline = weak_learner(self.classes)
	
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
		results = zip(*results)

		#print "Results: ",results	
		def avg(x):
			return sum(x)/len(x)
		return (map(avg, zip(*results[0])), map(avg, zip(*results[1])))
	
	def run_test(self, data):
		performance_booster = []
		performance_baseline = []
		baseline_correct = 0.0
		for (X,Y) in data:	
			if self.predictor.classify(X) == Y:
				self.correct += 1
			self.predictor.update(X,Y)
			self.t += 1
			performance_booster.append(self.correct / self.t)
			if self.baseline.predict(X) == Y:
				baseline_correct += 1
			self.baseline.partial_fit(X,Y)
			performance_baseline.append(baseline_correct / self.t)
		return performance_booster, performance_baseline

	def final_test(self,X,y,m):
		#print "Performing Test for",len(self.predictor)," weak learners.."
		correct_test = 0.0
		num_samples = 0.0
		data = zip(X,y)
		correct_baseline = 0.0
		for (X,y) in data:
			if self.predictor.classify(X) == y:
				correct_test +=1
			num_samples +=1
			if self.baseline.predict(X) == y:
				correct_baseline +=1
		print correct_test, correct_baseline
		avg = float(correct_test)/float(num_samples)
		baseline_avg = float(correct_baseline)/float(num_samples)
		return (avg,baseline_avg)
