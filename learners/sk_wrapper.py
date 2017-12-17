import numpy as np


class Wrapper(object):

    def predict(self, x):
        x = x.reshape(1,-1)
        try:
            return self.model.predict(x)[0]
        except:
            return self.classes[0]

    def partial_fit(self, x, y, sample_weight=1.0):
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        w = np.array([sample_weight])
        self.model.partial_fit( x, y, classes=self.classes, sample_weight=w)
