#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


mall = pd.read_csv('mall.csv.csv')


# In[3]:


mall.head()


# In[4]:


mall.info()


# In[5]:


mall.isna().any()


# In[6]:


mall.describe()


# In[8]:


mall.drop_duplicates(inplace = True)


# In[9]:


mall.head(30)


# In[32]:


x = mall.iloc[:,[3,4]].values


# In[33]:


from sklearn.cluster import KMeans


# In[35]:


wcss = []

for i in range (1,11):
    kmeans = KMeans (n_clusters = i ,init = 'k-means++' , random_state = 42)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)


# In[39]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker='o',color='red')
plt.title('the elboe method')
plt.ylabel('wcss')
plt.xlabel('number of clusters')


# In[41]:


kmeans = KMeans(n_clusters = 5 , init='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(x)


# In[44]:


plt.figure(figsize=(15,7))
sns.scatterplot(x[y_kmeans == 0,0],x[y_kmeans == 0,1],color='red',label='cluster 1')
sns.scatterplot(x[y_kmeans == 1,0],x[y_kmeans == 1,1],color='blue',label='cluster 2')
sns.scatterplot(x[y_kmeans == 2,0],x[y_kmeans == 2,1],color='green',label='cluster 3')
sns.scatterplot(x[y_kmeans == 3,0],x[y_kmeans == 3,1],color='yellow',label='cluster 4')
sns.scatterplot(x[y_kmeans == 4,0],x[y_kmeans == 4,1],color='purple',label='cluster 5')
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='gray',label='centroide',s=300)


# In[ ]:




