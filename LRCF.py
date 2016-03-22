import numpy as np

def linearRegCostFunction(X,y,theta,Lambda):
    y=y.flatten()
    theta=theta.flatten()
    m=X.shape[0]

    grad=np.zeros(theta.shape)

    J=1.0/(2*m)*((np.dot(X,theta)-y)**2).sum()+1.0*Lambda/(2*m)*(theta[1:]**2).sum()
    grad[0]=1.0/m*((np.dot(X,theta)-y)*X[:,0]).sum()
    aux1=np.dot(X,theta)-y
    aux1=aux1.reshape(aux1.shape[0],1)
    grad[1:]=1.0/m*sum(aux1*X[:,1:])+1.0*Lambda/m*theta[1:]
    return J
