import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import LRCF
import LRGradient
import TLR
import LC
import PF
import FN
import PFit
import VC

#Part One: Loading and Visualizing Data
print 'One: =========Loading and Visualizing Data...'
data=sio.loadmat('ex5data1')
X=data['X']
y=data['y']



m=X.shape[0]
plt.plot(X,y,'rx',linewidth=2,markeredgewidth=1)
plt.xlabel('Change in water level(x)')
plt.ylabel('Water flowing out of the dam(y)')
plt.title('Figure 1: Data')
plt.show()

print ''

#Part Two: Regularized Linear Regression Cost
print 'Two: ==========Regularized Linear Regression Cost...'
theta=np.array([[1],[1]])
tmpX=np.hstack((np.ones((m,1)),X))

J=LRCF.linearRegCostFunction(tmpX,y,theta,1)
print 'Cost at theta=[1;1] : %8.4f \n(This value should be about 303.993192)' % J
Theta=LRGradient.linearRegGradient(tmpX,y,theta,1)
print 'Gradient at theta=[1;1]: [%7.4f ; %7.4f] \n(this value should be about [-15.303016; 598.250744])' % (Theta[0],Theta[1])

#Part Three: Train Linear Regression
print 'Three: =============Train Linear Regression...'
Lambda=0
theta=TLR.trainLinearReg(tmpX,y,Lambda)

plt.plot(X,y,'rx',linewidth=2,markeredgewidth=1)
plt.xlabel('Change in water level(x)')
plt.ylabel('Water flowing out of the dam(y)')
plt.plot(X,np.dot(tmpX,theta),'--',lineWidth=2)
plt.show()

#Part Four: Learning Curve for linear Regression
print 'Four: ===================Learning Curve for Linear Regression...'
Xval=data['Xval']
yval=data['yval']
Lambda=100
tmp1=np.hstack((np.ones((m,1)),X))
tmp2=np.hstack((np.ones((Xval.shape[0],1)),Xval))

result=LC.learningCurve(tmp1,y,tmp2,yval,Lambda)
l1=plt.plot(np.array(range(m))+1,result[0],'r',np.array(range(m))+1,result[1],'b')
plt.title('Learning curve for linear regression')
plt.legend(l1,('Train','Cross Validataion'),loc=1)
plt.xlabel('Number of training examples')

plt.ylabel('Error')
plt.show()


#Part Five: Feature Mapping for Polynomial Regression
print 'Five: ==========================Feature Mapping for Polynomial Regression...'
p=8
X_poly=PF.polyFeatures(X,p)

result=FN.featureNormalize(X_poly)
X_poly=result[0]
mu=result[1]
sigma=result[2]

X_poly=np.hstack((np.ones((m,1)),X_poly))
Xtest=data['Xtest']

X_poly_test=PF.polyFeatures(Xtest,p)

X_poly_test=X_poly_test-mu
X_poly_test=X_poly_test/sigma
X_poly_text=np.hstack((np.ones((X_poly_test.shape[0],1)),X_poly_test))

X_poly_val=PF.polyFeatures(Xval,p)

X_poly_val=X_poly_val-mu
X_poly_val=X_poly_val/sigma
X_poly_val=np.hstack((np.ones((X_poly_val.shape[0],1)),X_poly_val))
print 'Normalized Training Example 1: ',X_poly[0,:]


#Part Six: Learning Curve for polynomial Regression
print 'Six: =============================Learning Curve for Polynomial Regression...'
Lambda=0.05

theta=TLR.trainLinearReg(X_poly,y,Lambda)

plt.plot(X,y,'rx',linewidth=2,markeredgewidth=1)
PFit.plotFit(X.min(axis=0),X.max(axis=0),mu,sigma,theta,p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (Lambda = '+str(Lambda)+' )')
plt.axis(xmin=-80,xmax=80,ymin=-60,ymax=40)
plt.show()

result=LC.learningCurve(X_poly,y,X_poly_val,yval,Lambda)
xaxis=np.array(range(m))+1
l1=plt.plot(xaxis,result[0],xaxis,result[1])
plt.title('Polynomial Regression Learning Curve (Lambda = '+str(Lambda)+' )')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis(xmin=0,xmax=13,ymin=0,ymax=100)
plt.legend(l1,('Train','Cross Validation'),loc=1)
plt.show()


#Part Seven: Validation for Selecting Lambda
print 'Seven: ======================= Validation for Selecting Lambda...'
result=VC.validationCurve(X_poly,y,X_poly_val,yval)
lambda_vec=result[0]
error_train=result[1]
error_val=result[2]
l1=plt.plot(lambda_vec,error_train,lambda_vec,error_val)
plt.legend(l1,('Train','Cross Validation'),loc=1)
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.axis(xmin=0,xmax=10,ymin=0,ymax=20)
plt.show()
























