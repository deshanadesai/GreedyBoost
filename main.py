import argparse
from random import seed
from sklearn import datasets
import test
from sklearn.datasets import load_svmlight_file
#warnings.filterwarnings("ignore", module="sklearn")

if __name__ == "__main__":
	seed(0)
	parser = argparse.ArgumentParser()

	#parser.add_argument('dataset', help='dataset filename')
	parser.add_argument('M', metavar='# weak_learners', help='number of weak learners', type=int)
	parser.add_argument('trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
	args = parser.parse_args()

	X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
	data = zip(X, y)

	accuracy = test.test(X,y, args.M, trials=args.trials)

	print "Accuracy:"
	print accuracy
	print "Baseline:"
	print baseline[-1]
