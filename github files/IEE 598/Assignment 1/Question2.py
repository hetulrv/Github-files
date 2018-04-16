# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib inline
plt.style.use('seaborn-white')
# IMPORTING THE DATA 

hp = pd.read_csv('K:/ASU docs/courses related docs/IEE598/Assignment/1/houseprice.csv')
#hp.info()
#hp.head(20)
df = (hp[hp.columns[:13]])
y = (hp[hp.columns[13]])


#1. CORRELATION PLOT BETWEEN THE VARIABLES

from string import ascii_letters
sns.set(style="white")

corr = hp.corr()
#print(corr)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(13, 13))
#print(df)
cmap = sns.diverging_palette(225, 12, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show(sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}))

#sns.pairplot(hp);

# fig, ax = plt.subplots(figsize=(10, 10))
# corr = hp.corr()
# sns.heatmap(corr, ax=ax)
# plt.savefig('K:/ASU docs/courses related docs/IEE598/Assignment/1/plot.jpeg')
    


#print(df)
#print(y)

#2. SIMPLE LINEAR REGRESSION FOR SEPERATE MODELS:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#lm = linear_model.LinearRegression()

#regr = skl_lm.LinearRegression()
#print(regr.intercept_)

#print(regr.coef_)



accuracy = 0
for i in X_train.columns:
    
#     print(len(y_train)== len(X_train[i])
#     print(X_train[i].shape, y_train.shape)
    regr = skl_lm.LinearRegression()
    #print(regr.intercept_)
    #print(regr.coef_)

    model = regr.fit(X_train[i].reshape((len(X_train[i]),1)), y_train)
    #print(X_train[i].reshape((len(X_train[i]))))
    predictions = regr.predict(X_test[i].reshape(len(X_test[i]),1))
#    predictions[0:5]
    print(i)
    plt.xlabel(i)
    plt.ylabel('y')
  
  #  print('Score:', model.score(X_test[i].reshape(len(X_test[i]),1), y_test))
    
    #RESIDUAL SUM OF SQUARES
    
    RSS = np.sum((y_test - regr.predict(X_test[i].reshape(len(X_test[i]),1)))**2)
    print('RSS is', RSS )
    MSE = RSS/len(y_test)
    print('MSE is ', MSE)
    plt.scatter(X_train[i].reshape((len(X_train[i]),1)), y_train)
    plt.plot(X_train[i].reshape((len(X_train[i]),1)), X_train[i].reshape(len(X_train[i]),1)*regr.coef_ + regr.intercept_,'red')
    plt.show()
    #plt.savefig('K:/plot'+i+'.png')
    
#3. MULTIPLE LINEAR REGRESSION:

from sklearn.model_selection import train_test_split
X = hp[hp.columns[:13]]
y = hp[hp.columns[13]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
regr = skl_lm.LinearRegression()
regr.fit(X_train,y_train)
#print(regr.intercept_)
#print(regr.coef_)
predictions = regr.predict(X_test)
#print(predictions)
RSS = np.sum((y_test - regr.predict(X_test))**2)
MSE = RSS/len(y_test)
print('The RSS for the Multiple Linear Regression is %f ' %RSS)
print('The MSE for the Multiple Linear Regression is %f ' %MSE)