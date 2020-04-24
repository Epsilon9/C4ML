import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h = sigmoid(np.dot(X,theta))
    
    J = -1/m*((np.log(h)).dot(y).T+(np.log(1-h)).dot(1-y).T)
    
    if np.isnan(J[0]):
       return(np.inf)
    return(J[0])


def gradient(theta, X, y):
    ''' calculate gradient descent for logistic regression
    '''
    m = y.size
    theta=theta.reshape(-1,1)
    h = sigmoid(np.dot(X,theta)) 
    
    grad = 1/m*(np.dot(X.T,h))

    return(grad.flatten())			# returns copy of array in one dimension