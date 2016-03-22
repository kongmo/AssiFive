import numpy as np

def polyFeatures(X,p):
    X_poly=np.zeros((X.size,p))
    shape=X.shape
    X=X.transpose()
    X=X.reshape(shape)
    for i in range(p):
        X_poly[:,i]=(X**(1+i)).flatten()
    return X_poly
