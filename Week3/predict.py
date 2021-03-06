import numpy as np
from sigmoid import sigmoid
def predict(theta, X, threshold=0.5):
  '''returns predicted value for X, given theta and threshold'''
  p = sigmoid(np.dot(X,theta.T)) >= threshold
  return(p.astype('int'))