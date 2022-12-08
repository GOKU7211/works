#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


ipl = pd.read_csv('matches.csv')


# In[3]:


ipl.head()


# In[4]:


ipl.shape


# In[5]:


ipl['player_of_match'].value_counts()


# In[6]:


ipl['player_of_match'].value_counts()[0:5]


# In[7]:


plt.figure(figsize=(10,10))
plt.bar(list(ipl['player_of_match'].value_counts()[0:5].keys()),list(ipl['player_of_match'].value_counts()[0:5]))


# In[8]:


ipl['result'].value_counts()


# In[9]:


ipl.columns


# In[10]:


ipl['toss_winner'].value_counts()


# In[11]:


bat= ipl[ipl['win_by_runs']!=0]


# In[12]:


bat.head()


# In[13]:


plt.figure(figsize=(6,6))
plt.hist(bat['win_by_runs'])
plt.show()


# In[20]:


#Extracting the records where a team won batting first
batting_first=ipl[ipl['win_by_runs']!=0]


# In[21]:


plt.figure(figsize=(7,7))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[22]:


#extracting those records where a team has won after batting second
batting_second=ipl[ipl['win_by_wickets']!=0]


# In[23]:


#looking at the head
batting_second.head()


# In[24]:


#Making a histogram for frequency of wins w.r.t number of wickets
plt.figure(figsize=(7,7))
plt.hist(batting_second['win_by_wickets'],bins=30)
plt.show()


# In[25]:


##Finding out the frequency of number of wins w.r.t each time after batting second
batting_second['winner'].value_counts()


# In[26]:


plt.figure(figsize=(7,7))
plt.bar(list(batting_second['winner'].value_counts()[0:3].keys()),list(batting_second['winner'].value_counts()[0:3]),color=["blue","green","orange"])
plt.show()


# In[27]:


#Making a pie chart for distribution of most wins after batting second
plt.figure(figsize=(7,7))
plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[28]:


#Looking at the number of matches played each season
ipl['season'].value_counts()


# In[29]:


#Looking at the number of matches played in each city
ipl['city'].value_counts()


# In[30]:


#Finding out how many times a team has won the match after winning the toss
import numpy as np
np.sum(ipl['toss_winner']==ipl['winner'])


# In[31]:


325/636


# In[ ]:




