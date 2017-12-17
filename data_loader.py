import numpy as np
from sklearn import datasets


def load_data(dataset):
	if dataset =="car":
		data = np.genfromtxt("data/car.csv",delimiter=",")
		rows,cols = data.shape
		X,y = data[1:,1:cols-1],data[1:,cols-1]
		X,y = X.astype(int), y.astype(int)

	elif dataset == "heart":
		data = np.genfromtxt("data/heart.csv",delimiter=",")
		rows,cols = data.shape
		X,y = data[1:,0:cols-1],data[1:,cols-1]
		y = np.where(y>0, 1,-1)
		y = y.astype(int)

	elif dataset == "ionosphere":
		data = np.genfromtxt("data/ionosphere.csv",delimiter=",")
		rows,cols = data.shape
		X,y = data[1:,0:cols-1],data[1:,cols-1]
		y = y.astype(int)
        
	elif dataset =="mushrooms":
		data = np.genfromtxt("data/processed_mushrooms.csv",delimiter=",")
		rows,cols = data.shape
		X,y = data[1:,2:cols],data[1:,1]
 		y_clean = []
		for item in y:
			if item==1:
				y_clean.append(item)
			elif item==0:
				y_clean.append(-1)
		y = np.array(y_clean)
 		y = y.astype(int)

	elif dataset =="cancer":
		data = datasets.load_breast_cancer()
		X = data.data
		y = data.target
		y_clean = []
		for item in y:
			if item==1:
				y_clean.append(item)
			elif item==0:
				y_clean.append(-1)
		y = np.array(y_clean)
        
	elif dataset == "digits":
		data = datasets.load_digits()
		X = data.data
		y = data.target  

	elif dataset == "iris":
		data = datasets.load_iris()
		X = data.data
		y = data.target  

	else:
		print "Dataset not found."
		X = []
		y = []
	return X,y