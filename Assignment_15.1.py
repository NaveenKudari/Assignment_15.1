
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
boston = load_boston()
X = pd.DataFrame(boston.data,columns=boston.feature_names)
Y=pd.DataFrame(boston.target)
lm=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=4)
lm.fit(x_train,y_train)
#Estimated intercept 
lm.intercept_
#the cofficients
lm.coef_
lm.predict(x_test)
print("mean squared error:%2f" % np.mean((lm.predict(x_test)-y_test)**2))
print("variance score is :%.2f" %lm.score(x_test,y_test))
plt.scatter(lm.predict(x_test),y_test)
plt.show()

