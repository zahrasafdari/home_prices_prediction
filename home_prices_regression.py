#!/usr/bin/env python
# coding: utf-8

# In[294]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[295]:


df = pd.read_csv("home_price.csv")

# take a look at the dataset
df.head()


# In[296]:


cdf=df[['Area','Room','Price']]


# In[297]:


df.dropna(inplace = True)


# In[298]:


df['Area'] = pd.to_numeric(df['Area'], errors='coerce')


# In[299]:


threshold = 300

# Filter the dataframe to remove rows where the 'area' value is above the threshold
df = df[df['Area'] <= threshold]


# In[300]:


df.shape


# In[313]:


msk = np.random.rand(len(df)) < 0.8 #test data of point 0.8
train = df[msk]
test = df[~msk]


# In[317]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['Area']])
y = np.asanyarray(train[['Price']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)



# In[318]:


y_hat= regr.predict(test[['Area']])
x = np.asanyarray(test[['Area']])
y = np.asanyarray(test[['Price']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[324]:


plt.scatter(train.Area, train.Price,  color='blue')
XX = np.arange(0.0, 300.0, 0.1)
yy = regr.intercept_[0]+ regr.coef_[0][0]*XX
plt.plot(XX, yy, '--r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[ ]:





# In[ ]:





# In[ ]:




