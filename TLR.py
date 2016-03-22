from scipy.optimize import fmin
import scipy.optimize as sop
import numpy as np
import LRCF

def trainLinearReg(X,y,Lambda):
    initial_theta=np.zeros((X.shape[1],1))
    costFunc=lambda x:LRCF.linearRegCostFunction(X,y,x,Lambda)
    theta=sop.minimize(costFunc,initial_theta,method='BFGS')
    theta=theta.x
    #theta=fmin(costFunc,initial_theta)
    return theta
