import matplotlib.pyplot as plt
import PF
import numpy as np

def plotFit(min_x,max_x,mu,sigma,theta,p):
    x=(np.arange(min_x-15,max_x+25,0.05))
    X_poly=PF.polyFeatures(x,p)
    X_poly=X_poly-mu
    X_poly=X_poly/sigma

    X_poly=np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
    plt.plot(x,np.dot(X_poly,theta),'--',linewidth=2)
                    
