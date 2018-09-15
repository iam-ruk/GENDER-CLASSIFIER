import numpy as np
from training_set import *
W1 = np.random.randn(3,3)*0.01
W2 = np.random.randn(1,3)*0.01
B1 = np.zeros((3,1))
B2 = np.zeros((1,1))
print(X.shape)
print(Y.shape)

def sigmoid(x):
    return 1/(1 + np.exp(-x))



Z1 = np.dot(W1, X.T) + B1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + B2
A2 = sigmoid(Z2)
dz2 = A2 - Y
dw2 = 1 / 55 * np.dot(dz2, A1.T)
dz1 = np.multiply(np.dot(W2.T, dz2), np.multiply(A1, 1 - A1))
dw1 = (1 / 55) * np.dot(dz1, X)
db2 = (1 / 55) * np.sum(dz2, axis=1, keepdims=True)
db1 = (1 / 55) * np.sum(dz1, axis=1, keepdims=True)


def cost(y,y1):
    return -1/55*(np.sum(np.multiply(y,np.log(y1))+np.multiply(1-y,np.log(1-y1))))

k=100
for i in range(1,14000):
    W1 = W1 - 0.69*dw1
    W2 = W2 - 0.95*dw2
    B1 = B1 - 0.52*db1
    B2 = B2 - 0.03*db2
    Z1 = np.dot(W1, X.T) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    dz2 = A2 - Y
    dw2 = 1 / 55 * np.dot(dz2, A1.T)
    dz1 = np.multiply(np.dot(W2.T, dz2), np.multiply(A1, 1 - A1))
    dw1 = (1 / 55) * np.dot(dz1, X)
    db2 = (1 / 55) * np.sum(dz2, axis=1, keepdims=True)
    db1 = (1 / 55) * np.sum(dz1, axis=1, keepdims=True)
    if(i % 100 == 0):
        print("cost after " +str(k)+ "iterations is" + str(cost(Y,A2)))
        k+=100
name=input("Enter your NAME")
first_letter=parameter(name[0])
last_letter=parameter(name[-1])
length=len(name)
def predict(fl,ll,len):
    Z1 = np.dot(W1,[[fl],[ll],[len]]) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return np.round(A2)
prediction=predict(first_letter,last_letter,length)
if(np.squeeze(prediction)==0):
    print("Boy")
else:
    print("GAL")
