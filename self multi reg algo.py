#!/usr/bin/env python
# coding: utf-8

# In[148]:


# import necessory library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[150]:


# load data
df = pd.read_csv("D:/Machine Learning/datasets/Housing full.csv")
# price/lotsize/bedrooms/bathrms/stories/driveway/recroom/fullbase/gashw/airco/garagepl/prefarea


# In[151]:


def normalize(data):
#   normalize by data = (x-mean of data) / stddev of data
#   here data is our features(X)
    mean = np.mean(data)
    std = np.std(data)
    data = ( data - mean ) / std
#     return mean and std because it needed in predicting values
    return data,mean,std


# In[152]:


# cost functoin
def calc_cost(x,y,w):
    return round( ( (np.sum( (y - np.sum((w*x),axis=1))**2 ) ) / (2*len(y)) ),3 )
# calc_cost(X,Y,[1,1,1,1,1,1,1,1,1,1,1,1])


# In[153]:


# gradeint decent algorithm
# here alpha = vecor of weights
# theta = learning rate
# iters = number of steps
def gradient_decent(x,y,alpha,theta,iters):
    for i in range(iters):
        theta = gradient_step(x,y,alpha,theta)
    err = calc_cost(x,y,theta)
    return theta , err


# In[154]:


# now implement actual algorithm
def gradient_step(x,y,alpha,w):
    m = len(y)
    pred = np.sum((w*x),axis = 1)
    pred = y - pred 
    deriv = np.sum( ((-x)*pred[:,np.newaxis]),axis=0 )
    w = w - (alpha * (deriv/m) )
    return w


# In[155]:


def predict(x,theta,mean,std):
#     first of all normalize x (test set) because we train our data at normalized scale
    x = np.array((x - mean) / std)
#     insert 1 at 0th index
    x = np.insert(x,0,1)
#     return predicted value
    return np.sum(x*theta)


# In[159]:


# saparate features and target
X = df.iloc[:,1:]
Y = df.iloc[:,0]
# nomrlize data
X, mean, std = normalize(X)

# attach ones to X for  X0
ones = np.ones((len(X),1),dtype=int)
X = np.concatenate((ones,X),axis=1)


# In[168]:


# initialize theta to 0 or any value
theta = np.zeros(X.shape[1])
# run gradidnt decent
theta , err = gradient_decent(X,Y,0.01,theta,1000)
print("weights : ",theta,"\n\nmean squared err : ",err)


# In[160]:


# predict price with these features
p = [2000,4,3,4,1,1,1,1,1,1,1]
pred = predict(p,theta,mean,std)
print(pred)

