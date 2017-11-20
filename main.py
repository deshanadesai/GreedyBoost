import argparse
from random import seed

if __name__ == "__main__":
	seed(0)
	parser.add_argument('dataset', help='dataset filename')
	parser.add_argument('M', metavar='# weak_learners', help='number of weak learners', type=int)
	parser.add_argument('trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
	args = parser.parse_args()

	X, y = load_svmlight_file(args.dataset)
	data = zip(X, y)

	accuracy, baseline = test(data, args.M, trials=args.trials)

	print "Accuracy:"
	print accuracy
	print "Baseline:"
	print baseline[-1]
