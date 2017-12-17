import argparse
from random import seed
from sklearn import datasets
import test
from sklearn.datasets import load_svmlight_file
import numpy as np
import data_loader
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ensembles import osboost, ocpboost, expboost, ozaboost, smoothboost
from sklearn.model_selection import train_test_split
from learners import nb_gaussian,sk_nb, nb, perceptron, sk_perceptron, random_stump, sk_decisiontree, decision_trees
import math
from tqdm import *

if __name__ == "__main__":
    seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help='dataset filename')
    parser.add_argument('weak_learner', help='chosen weak learner')
    parser.add_argument('M', metavar='# weak_learners', help='number of weak learners', type=int)
    parser.add_argument('trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
    parser.add_argument('--record', action='store_const',const=True, default = False, help = 'export the results in file')
    args = parser.parse_args()

    X,y = data_loader.load_data(args.dataset)

    algorithms = {
    "smoothboost":smoothboost.SmoothBoost,
    "osboost":osboost.OSBoost,
    "ocpboost":ocpboost.OCPBoost,
    "expboost":expboost.EXPBoost,
    "ozaboost":ozaboost.OzaBoostClassifier}

    weak_learners ={
    "sk_perceptron": sk_perceptron.PerceptronClassifier,
    "sk_nb":sk_nb.NaiveBayes,
    "gaussian_nb":nb_gaussian.NaiveBayes,
    "nb": nb.NaiveBayes,
    "sk_decisiontree": sk_decisiontree.DecisionTree,
    "random_stump":random_stump.RandomStump,
    "decisiontree":decision_trees.DecisionTree,
    "perceptron":perceptron.Perceptron}

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1)


    T = 20
    fig, ax = plt.subplots( nrows=1, ncols=1 )  


    for (name, ensembler) in tqdm(algorithms.iteritems()):
        print "Ensembler: ",name
        num_samples = []
        results = []
        for i in range(1,11):
            num = int(math.floor(i/10.0*X_train.shape[0]))
            print "Number of samples: ",num, X_train.shape
            num_samples.append(num)
            acc = 0.0
            for t in range(T):
                X_cut = X_train[0:num]
                y_cut = y_train[0:num]
                model = test.Test(ensembler, weak_learners[args.weak_learner],X_cut,y_cut,args.M)
                train_accuracy, baseline_train_accuracy = model.test(X_cut,y_cut,X_test, y_test, args.M,trials=args.trials)
                test_accuracy, baseline_test_accuracy = model.final_test(X_test,y_test,args.M)
                acc += test_accuracy
            results.append(acc/float(T))
        print "Result: ",results
        ax.plot(num_samples, results, label = name)
        fig.savefig('Ensemblers_num_samples.png')


    plt.legend(loc="upper left")
    plt.xlabel('Number of Samples', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Ensemblers accuracy vs Number of samples', fontsize = 16)
    fig.savefig('Ensemblers_num_samples.png')
    plt.close(fig)


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
