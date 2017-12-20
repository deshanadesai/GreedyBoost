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
from learners import nb_gaussian,sk_nb, nb, perceptron, random_stump, sk_decisiontree, decision_trees, sk_perceptron
from tqdm import *

class OnlineLearners():
    def __init__(self):
        self.learner = None
        self.run()
        
    def run(self):
        seed(0)
        parser = argparse.ArgumentParser()

        parser.add_argument('dataset', help='dataset filename')
        args = parser.parse_args()

        weak_learners ={
        "Perceptron":perceptron.Perceptron,
        "Naive Bayes":sk_nb.NaiveBayes,
        "Decision Stumps":random_stump.RandomStump}

        heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
        rounds = 10
        X, Y = data_loader.load_data(args.dataset)

        x_axis = 1. - np.array(heldout)
        fig, ax = plt.subplots( nrows=1, ncols=1 )  


        for (name, clf) in weak_learners.iteritems():
            print("training %s" % name)
            rng = np.random.RandomState(42)
            y_axis = []
            for i in heldout:
                yy_ = []
                for r in range(rounds):
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=rng)
                    data = zip(X_train, y_train)
                    classes = np.unique(np.array([y for (x,y) in data]))
                    self.learner = clf(classes)
                    
                    for (x,y) in data:
                        self.learner.partial_fit(x, y)
                    
                    test_data = zip(X_test, y_test)
                    incorrect = 0.0
                    for (x,y) in test_data:
                        y_pred = self.learner.predict(x)
                        if y_pred != y:
                            incorrect+=1.0
                    yy_.append(incorrect/X_test.shape[0])
                y_axis.append(np.mean(yy_))
            ax.plot(x_axis, y_axis, label=name)

        plt.legend(loc="upper right")
        plt.xlabel("Proportion train")
        plt.ylabel("Test Error Rate")
        plt.show()
        fig.savefig('Compare_online_learners.png')
        plt.close(fig)

OnlineLearners()