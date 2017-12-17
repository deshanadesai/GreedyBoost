import numpy as np
from sklearn.linear_model import Perceptron


class PerceptronClassifier(object):
    def __init__(self, classes):
        self.classes = classes
        self.model = Perceptron()
        self.w = None
        
    def predict(self,X):
        X = X.reshape(1,-1)
        try:
            return self.model.predict(X)[0]
        except:
            return self.classes[0]
    
    def partial_fit(self, X, y, sample_weight = 1.0):
        X = X.reshape(1,-1)
        y = y.reshape(1,-1)
        return self.model.partial_fit(X,y, sample_weight = [sample_weight], classes = self.classes)