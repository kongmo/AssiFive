import numpy as np
import TLR


def learningCurve(X,y,Xval,yval,Lambda):
    m=X.shape[0]
    error_train=np.zeros((m,1))
    error_val=np.zeros((m,1))

    n=Xval.shape[0]
    for i in range(m):
        ThetaTrain=TLR.trainLinearReg(X[0:(i+1),:],y[0:(i+1)],Lambda)
        t=X[0:(i+1),:].shape[0]
        error_train[i]=1.0/(2*t)*(((np.dot(X[0:(i+1),:],ThetaTrain)-y[0:(i+1)].flatten())**2).sum())
        error_val[i]=1.0/(2*n)*(((np.dot(Xval,ThetaTrain)-yval.flatten())**2).sum())
    return error_train,error_val
