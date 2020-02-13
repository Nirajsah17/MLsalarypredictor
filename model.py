# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:55:03 2020

@author: niraj sen
"""
# import all library used here :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 
hdataset = pd.read_csv('hiring.csv')
hdataset['experience'].fillna(0,inplace = True)
hdataset['test_score(out of 10)'].fillna(hdataset['test_score(out of 10)'].mean(),inplace = True)
X = hdataset.iloc[:,:3]
# converting words into integer value :
def convert_into_int(words):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5, 'six':6,'seven':7,'eight':8,'nine':9,
                 'ten':10, 'eleven':11, 'twelve':12,'thirteen':13,'zero':0,0:0}
    
    return word_dict[words]
X['experience'] = X['experience'].apply(lambda x :convert_into_int(x))
y1 = hdataset.iloc[:,-1]
# spliiting training and test sat:
# since we have very small datasets, we will train our models with all available data :
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y1)
# saving models to disk :
pickle.dump(reg,open('model.pkl','wb'))
# loading models to compare the results:
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))