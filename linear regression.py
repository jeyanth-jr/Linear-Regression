from ctypes import sizeof
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
data=np.genfromtxt(r'data.txt',delimiter=",")
X=data[:,0]
Y = data[:, 1].reshape(X.size, 1)
X = np.vstack((np.ones((X.size, )), X)).T
""" print(X.shape)
print(Y.shape) """
plt.scatter(X[:,1],Y)
plt.xlabel('area in Sq.ft')
plt.ylabel('Price')
plt.show()
J=0
def meannormalisation(X):
    mean=np.mean(X)
    std=np.std(X)
    for i in range(0,X.shape[0]):
        X[i,1]=(X[i,1]-mean)/std
    return X    
def model(X,Y,alpha,iter):
    m=Y.size
    J_list=[]
    theta=np.zeros((2,1))
    for i in range(iter):
        h_x=np.dot(X,theta)
        J=(np.sum(np.square(h_x-Y)))/(2*m)
        d_theta = (1/m)*np.dot(X.T, h_x - Y)
        theta=theta-alpha*d_theta
        J_list.append(J)
    return theta,J_list
alpha=0.00000005
iter=300
theta,J=model(X,Y,alpha,iter)
rng=np.arange(0,iter)
plt.plot(J,rng)
plt.subplot()
plt.xlabel('No of iterations')
plt.ylabel('Cost Function')
plt.show()
x=np.arange(0,2500)
n=[]
for i in range(0,2500):
    n.append((theta[1,0]*x[i])+theta[0,0])
plt.plot(x,n,'b')    
plt.scatter(X[:,1],Y)
plt.show()
