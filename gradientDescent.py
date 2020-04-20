import numpy as np
from computeCost import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    return theta updated
    """
    m=len(y)
    for i in range(num_iters):
        temp=np.dot(X,theta)-y
        temp = np.dot(X.T,temp)
        theta= theta-(alpha * 1/m * temp)
        if(i%100==0):
            print('Cost function after ',i,' iterations is ', computeCost(X,y,theta))
    
    return theta