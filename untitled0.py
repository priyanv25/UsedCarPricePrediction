# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:40:44 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({ 
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.bottom': False,
    'ytick.left': False
})


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('car.csv')
data.head()
data.info()
sns.pairplot(data)
h1=data.iloc[:,2].values
h2=data.iloc[:,4].values

plt.scatter(h2, h1, color = 'red')


plt.xlabel('kmdriven')
plt.ylabel('Selling price')
plt.show()

h3=data.iloc[:,1].values
h4=data.iloc[:,2].values
plt.scatter(h3, h4, color = 'red')


plt.xlabel('year')
plt.ylabel('Selling price')
plt.show()


data.Transmission.value_counts()
data.loc[:,['Transmission','Selling_Price']].sort_values(by=['Selling_Price'],ascending =False)['Transmission'].head(15).value_counts().plot.pie(figsize=(15,15),subplots=True, autopct='%.1f%%',explode=[0,.08],shadow=True)

le = LabelEncoder()
df = pd.get_dummies(data['Fuel_Type'],prefix='FT',drop_first=True)
data['Seller_Type'] = le.fit_transform(data['Seller_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])
data = pd.concat([data,df],axis=1)
data.drop(['Fuel_Type'],axis=1,inplace=True)
data.head()
def gen_model(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    lr = LinearRegression()
    
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    
    coeffecients = pd.DataFrame(lr.coef_,X.columns)
    coeffecients.columns = ['Coeffecient']
    print('Coefficients :  {coeffecients} ')
    
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error of Test Set : {mse}')
    print('Root Mean Square Error of Test Set : {rmse}')
    
    yt_pred = lr.predict(X_train)
    tmse = mean_squared_error(y_test,y_pred)
    trmse = np.sqrt(mse)
    print('Mean Squared Error of Train Set : {tmse}')
    print('Root Mean Square Error of Train Set : {trmse}')
    

    fig,ax1 = plt.subplots(figsize=(15,8))
    fig = sns.scatterplot(y_test,y_pred,ax=ax1)
    plt.xlabel('Y true')
    plt.ylabel('Y predicted')
    plt.title('True vs Predicted')
    plt.show(fig)
    

A= data.drop(['Car_Name','Selling_Price'],axis = 1)
B = data['Selling_Price']
gen_model(A,B)