import numpy as np
import TLR

def validationCurve(X,y,Xval,yval):
    lambda_vec=np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]).transpose()

    error_train=np.zeros((lambda_vec.size,1))
    error_val=np.zeros((lambda_vec.size,1))

    m=X.shape[0]
    n=Xval.shape[0]
    for i in range(lambda_vec.size):
        thetaTrain=TLR.trainLinearReg(X,y,lambda_vec[i])
        error_train[i]=1.0/(2*m)*(((np.dot(X,thetaTrain)-y.flatten())**2).sum())
        error_val[i]=1.0/(2*n)*(((np.dot(Xval,thetaTrain)-yval.flatten())**2).sum())
    return lambda_vec,error_train,error_val
