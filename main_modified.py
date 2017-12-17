import argparse
from random import seed
from sklearn import datasets
from modifications import test_dynamic, ozaboost_dynamic
import numpy as np
import data_loader
from ensembles import osboost, ocpboost, expboost, ozaboost, smoothboost
from sklearn.model_selection import train_test_split
from learners import nb_gaussian,sk_nb, nb, perceptron, random_stump, sk_decisiontree, decision_trees
from tqdm import *

if __name__ == "__main__":
	seed(0)
	parser = argparse.ArgumentParser()

	parser.add_argument('dataset', help='dataset filename')
	parser.add_argument('algorithm', help = 'Boosting algorithm')
	parser.add_argument('weak_learner', help='chosen weak learner')
	parser.add_argument('M', metavar='# weak_learners', help='number of weak learners', type=int)
	parser.add_argument('trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
	parser.add_argument('--record', action='store_const',const=True, default = False, help = 'export the results in file')
	args = parser.parse_args()


	X,y = data_loader.load_data(args.dataset)

	algorithms = {
	"ozaboost":ozaboost_dynamic.OzaBoostClassifier}

	weak_learners ={
	"sk_nb":sk_nb.NaiveBayes,
	"gaussian_nb":nb_gaussian.NaiveBayes,
	"nb": nb.NaiveBayes,
	"sk_decisiontree": sk_decisiontree.DecisionTree,
	"random_stump":random_stump.RandomStump,
	"decisiontree":decision_trees.DecisionTree,
	"perceptron":perceptron.Perceptron}

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1)

	results = []
	T = 10
    
	for t in tqdm(range(T)):
		model = test_dynamic.Test(algorithms[args.algorithm], weak_learners[args.weak_learner],X_train,y_train,args.M)
		train_accuracy, baseline_train_accuracy = model.test(weak_learners[args.weak_learner], X_train,y_train,X_test, y_test, args.M,trials=args.trials)
		test_accuracy, baseline_test_accuracy = model.final_test(X_test,y_test,args.M)
		results.append(test_accuracy)
	print float(sum(results))/10.0
        
        
	#print "Running for "+str(args.M)+" weak learners.."
	#print "Shape of Train Data: ",X_train.shape,y_train.shape
	#print "Shape of Test Data: ",X_test.shape, y_test.shape
	#test_accuracy, baseline_test_accuracy = model.final_test(X_test,y_test,args.M)
	#print "Accuracy on Training Set/ Baseline :"
	#print train_accuracy[-1], baseline_train_accuracy[-1]
	#print "Test Accuracy/ Baseline: "
	#print test_accuracy, baseline_test_accuracy
	
	
	if args.record:
		results = {
			'm': args.M,
			'Training Accuracy':train_accuracy,
			'Baseline Training Accuracy': baseline_train_accuracy,
			'Testing Accuracy': test_accuracy,
			'Baseline Testing Accuracy': baseline_test_accuracy,
			'trials': args.trials,
			'seed': 0
		}
		filename = open('adaboost_results_'+str(args.dataset)+'_'+str(args.M)+'.txt','a')
		filename.write("BEGIN\n")
		for k,v in results.items():
			filename.write(str(k)+"~~ "+str(v)+"\n")
		filename.write("END\n\n")
		filename.close()
