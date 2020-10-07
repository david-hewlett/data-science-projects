#!/usr/bin/env python
# coding: utf-8

# ** Introduction **
# 
# This project looks at communal bike rentals in the Washington D.C. area. The dataset includes information around the time of rental and the weather. Our goal of this project is to test different modles to see if we can predict the total number of bikes rented in a given hour (the 'cnt' column - which is a combination of the casual and registered columns). We will use Linear Regression, a Decision Tree and Random Forest to make our predictions. 

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
get_ipython().magic('matplotlib inline')


# In[2]:


bike_rentals = pd.read_csv('bike_rental_hour.csv')


# In[3]:


# Exploring the initial data

print(bike_rentals.head(10))


# In[4]:


# Gain insight on the 'cnt' column:

plt.hist(bike_rentals['cnt'])


# In[5]:


# Explore the correlations with the 'cnt' column:

cor_map = bike_rentals.corr()
print(abs(cor_map['cnt']).sort_values())


# In[6]:


# Change the hour variable to a time of day variable

def assign_label(hour):
    if 6 > hour <= 12:
        tod = 1
    elif 12 > hour <= 18:
        tod = 2
    elif 18 > hour <= 24:
        tod = 3
    else:
        tod = 4
    return tod

bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)


# In[7]:


# Re-look at the correlations with the time_label column.

cor_map = bike_rentals.corr()
print(abs(cor_map['cnt']).sort_values())


# ** ERROR METRIC **
# 
# We will use the RMSE metric to measure our data.
# 
# We will start with linear regression on the data for our prediction.

# In[8]:


# First we need to randomize and split up the data into train 
# and test groups.


# In[9]:


bike_rentals = bike_rentals.sample(frac=1).reset_index(drop=True)
split_num = math.floor(bike_rentals.shape[0] * .8)
train = bike_rentals.iloc[:split_num]
test = bike_rentals.iloc[split_num:]


# In[10]:


# Now we need to create a list of columns to use for our regression
# We will remove 'cnt', 'casual' and 'registered' because they would
# inform the algorithm of the answer. We will remove 'atemp'
# because it is a function of temp and we don't want to duplicate.
# We will also remove dteday. 

features = list(bike_rentals.drop(['cnt',
                               'casual',
                               'registered',
                               'atemp',
                               'dteday'], axis=1
                            ).columns)


# In[11]:


lr = LinearRegression()
lr.fit(train[features],train['cnt'])


# In[12]:


predictions = lr.predict(test[features])
lr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
print('Test RMSE: ',lr_rmse)


# ** Analysis of RMSE**
# 
# This RMSE seems high. This could be from the distribution of the data. Most of the rentals are low but some are extremely high and those may be getting penalized by the Linear Regression model.

# In[13]:


dtr = DecisionTreeRegressor()
dtr.fit(train[features],train['cnt'])

predictions = dtr.predict(test[features])
dt_rmse = mean_squared_error(test['cnt'], predictions)**(1/2)
print('Test RMSE: ', dt_rmse)


# The RMSE is extremely high pointing to severe overfitting. Let's try to tweak it a bit and find out what the best parameters and values would be to change. We'll start with min_samples and then test depth.

# In[14]:


for i in range(1,11):
    dtr = DecisionTreeRegressor(min_samples_leaf=i)
    dtr.fit(train[features],train['cnt'])
    
    predictions = dtr.predict(test[features])
    dt_rmse = mean_squared_error(test['cnt'], predictions)**(1/2)
    print('Test RMSE',i,'min_sample_leaf: ', dt_rmse)


# In[15]:


for i in range(10,36):
    dtr = DecisionTreeRegressor(max_depth=i, min_samples_leaf=5)
    dtr.fit(train[features],train['cnt'])
    
    predictions = dtr.predict(test[features])
    dt_rmse = mean_squared_error(test['cnt'], predictions)**(1/2)
    print('Test RMSE',i,'max_depth: ', dt_rmse)


# In[16]:


dtr = DecisionTreeRegressor(max_depth = 24,min_samples_leaf = 5)
dtr.fit(train[features],train['cnt'])
    
predictions = dtr.predict(test[features])
dt_rmse = mean_squared_error(test['cnt'], predictions)**(1/2)
print('Test RMSE max_depth = 24, min_samples_leaf = 5: ', dt_rmse)


# ** Quick Analysis **
# 
# After playing around with it a little, it looks like the optimal spot may be somewhere around 5 min sample leafs and 24 max depth. If we wanted to be even more sure we could run multiple iterations of this and take the average. With this data and these features we could get our RMSE down more easily with min_samples_leaf than max_depth. It is probably worth dropping max_depth to speed up the operation, especially if we were working with a larger data set.
# 
# Compared to Linear Regression, the Decision Tree Regressor produced a significantly better RMSE value. 
# 
# Next we try applying a Random Forest Regressor. We'll start with the default setting and then experiment with the parameters.

# In[21]:


rfr = RandomForestRegressor()
rfr.fit(train[features],train['cnt'])

predictions = rfr.predict(test[features])
rfr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
print('Test RMSE: ',rfr_rmse)


# We can already see that the random forest regressor is fitting better to the data than either the linear regression model or the decision tree model.

# In[24]:


for i in range(1,11):
    rfr = RandomForestRegressor(min_samples_leaf = i)
    rfr.fit(train[features],train['cnt'])
    
    predictions = rfr.predict(test[features])
    rfr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
    
    print('Test RMSE with min_sample_leaf =',i,':',rfr_rmse)
    


# In[27]:


for i in range(5,31):
    rfr = RandomForestRegressor(max_depth = i)
    rfr.fit(train[features],train['cnt'])
    
    predictions = rfr.predict(test[features])
    rfr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
    
    print('Test RMSE with max_depth =',i,':',rfr_rmse)


# In[28]:


for i in range(1,31):
    rfr = RandomForestRegressor(n_estimators=i)
    rfr.fit(train[features],train['cnt'])
    
    predictions = rfr.predict(test[features])
    rfr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
    
    print('Test RMSE with n_estimators =',i,':',rfr_rmse)


# Now that we have some idea of the optimal spots for our parameters lets try a combo.

# In[51]:


rmse_list = []

kf = KFold(n_splits=10,shuffle=True)
X = bike_rentals.index

for train_index, test_index in kf.split(X):
    train = bike_rentals.iloc[train_index]
    test = bike_rentals.iloc[test_index]
    
    rfr = RandomForestRegressor(max_depth = 19, n_estimators = 29)
    rfr.fit(train[features],train['cnt'])
    
    predictions = rfr.predict(test[features])
    rfr_rmse = mean_squared_error(test['cnt'],predictions)**(1/2)
    
    rmse_list.append(rfr_rmse)
    
    print('Test RMSE: ',rfr_rmse)
    
avg_rmse = np.mean(rmse_list)
print('')
print('Avg RMSE for Random Forest:', avg_rmse)


# ** Conclusion **
# 
# We were able to get our RMSE down from 140.0 inially with linear regression to 41.2 with our random forest model. This was a great example of how different models work and how to test them.

# In[ ]:




