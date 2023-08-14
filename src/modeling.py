from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import numpy as np

class LinearModel(LinearRegression):

    def __call__(self, X):
        
        if len(X.shape) == 1:
            return self.predict([X])
        else:
            return self.predict(X)
        
class QuadraticModel(LinearModel):

    def __init__(self):
        super().__init__()
        self.poly = PolynomialFeatures(degree=2)

    def __call__(self, X):
        return self.predict(X)

    def fit(self, X, y):
        X_stack = np.stack(X)
        poly_features = self.poly.fit_transform(X_stack)
        super().fit(poly_features, y)
        return self

    def predict(self, X):

        if isinstance(X, list):
            X_stack = np.stack(X)
        else:
            X_stack = X.reshape(1, -1)

        X_poly = self.poly.transform(X_stack)
        return super().predict(X_poly)
    
class RandomForestModel(RandomForestRegressor):

    def __call__(self, X):
        
        if len(X.shape) == 1:
            return self.predict([X])
        else:
            return self.predict(X)
        
class SVMModel():

    def __init__(self):
        self.svms = []

    def fit(self, X, y):
        num_svms = len(y[0])
        self.svms = [SVR().fit(X, [j[i] for j in y]) for i in range(num_svms)]

    def predict(self, X):
        
        if len(X.shape) == 1:
            return self.predict([X])
        
        else:
            return [svm.predict(X) for svm in self.svms]