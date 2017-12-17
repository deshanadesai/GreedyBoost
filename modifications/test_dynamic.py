from random import shuffle
from collections import defaultdict
import numpy as np
import ozaboost_dynamic

class Test():
	def __init__(self,algorithm, weak_learner, X, y, m):
		data = zip(X,y)
		self.classes = np.unique(np.array([y for (x,y) in data]))
		self.predictor = algorithm(learners = weak_learner, classes = self.classes, total_points = m)
		self.correct = 0.0
		self.t = 0
		self.baseline = weak_learner(self.classes)
	
	def test(self, learner, X,y,X_val, y_val, m, trials=1, should_shuffle=True):
		results = []
		data = zip(X,y)

		for t in range(trials):
			if should_shuffle:
				shuffle(data)
			results.append(self.run_test(learner, data))

		results = zip(*results)
		def avg(x):
			return sum(x)/len(x)
		return (map(avg, zip(*results[0])), map(avg, zip(*results[1])))
	
	def run_test(self, learner, data):
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
			
			if clf_no<100 and key!=Y and (conf<0.8 or conf==1.0):#  or (key==Y and conf<0.5):
				clf_no +=1
				#print ("Initializing new learner: ",clf_no)
				#print (label_weights)
				#print ("Initial confidence in incorrect classification",key," : ",conf, "& correct classification: ",Y)
				
				self.predictor.initialize_new_learner(learner, X,Y,weight, prev_points)
			self.t += 1
			performance_booster.append(self.correct / self.t)
			if self.baseline.predict(X) == Y:
				baseline_correct += 1
			self.baseline.partial_fit(X,Y)
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
			if self.baseline.predict(X) == y:
				correct_baseline +=1
		avg = float(correct_test)/float(num_samples)
		baseline_avg = float(correct_baseline)/float(num_samples)
		return (avg,baseline_avg)
