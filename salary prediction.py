#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


salary = pd.read_csv('salary.csv')


# In[3]:


salary.head()


# In[4]:


salary.describe()


# In[5]:


salary.isnull().any()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


plt.plot(salary['Salary'],salary['YearsExperience'])


# In[10]:


from sklearn.model_selection import train_test_split
x = salary.drop('Salary',axis = 1)


# In[11]:


x.head()


# In[12]:


y = salary['Salary']


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 42)


# In[16]:


from sklearn.linear_model import LinearRegression
l = LinearRegression()


# In[17]:


l.fit(x_train,y_train)


# In[18]:


ypredict = l.predict(x_test)


# In[21]:


y_test


# In[19]:


ypredict


# In[20]:


print(l.score(x_test,y_test))


# In[ ]:




