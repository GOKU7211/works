#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd 
import numpy as np


# In[111]:


stu = pd.read_csv('student-mat.csv')


# In[112]:


stu.head()


# In[113]:


stu.shape


# In[114]:


stu.columns


# In[115]:


stu['G3'].describe()


# In[116]:


stu.isnull().any()


# In[117]:


stu.isna().any()


# In[118]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[119]:


sns.countplot(stu['G3']).axes.set_title('Final grade of students',fontsize=20).axes.set_xlabel('final grade',fontsize = 10)


# length of male and female students

# In[120]:


## lenght of male students 
len(stu[stu['sex']=='M'])


# In[121]:


## lenth of female students
len(stu[stu['sex']=='F'])


# Average grade of student(G1,G2,G3)

# In[122]:


stu['avg'] = (stu['G1'] + stu['G2'] + stu['G3'])/3


# In[123]:


stu['avg'].head()


# In[124]:


def find_grade(stu):
    grade = []
    
    for row in stu['avg']:
        
        
        if row >= (.9 * stu['avg'].max()):
            grade.append('1')
            
        elif row >= (.7* stu['avg'].max()):
            grade.append('2')
            
        elif row < (.7 * stu['avg'].max()):
            grade.append('3')
            
    stu['avg'] = grade
    
    return stu
    
    


# In[125]:


data = find_grade(stu)


# In[126]:


data.drop(['school','age'],axis=1, inplace =True)


# In[127]:


data.head()


# In[128]:


data.columns


# In[129]:


## yes / no values

d = { 'yes':1,'no' : 0}


# In[130]:


data['schoolsup'] = data['schoolsup'].map(d)
data['famsup']    = data['famsup'].map(d)
data['paid']  = data['paid'].map(d)
data['activities'] = data['activities'].map(d)
data['nursery'] = data['nursery'].map(d)
data['higher'] = data['higher'].map(d)
data['internet'] = data['internet'].map(d)
data['romantic'] = data['romantic'].map(d)


# In[131]:


d ={'F':0,'M':1}


# In[132]:


data['sex'] = data['sex'].map(d)


# In[133]:


data['Mjob'].unique()


# In[134]:


d = { 'health':0,'services':1, 'teacher':2,'at_home':3,'other':4}


# In[135]:


data['Mjob'] = data['Mjob'].map(d)


# In[136]:


data['Fjob'] = data['Fjob'].map(d)


# In[137]:


d = { 'course':0,'home':1,'reputation':2,'other':3}


# In[138]:


data['reason'] = data['reason'].map(d)


# In[139]:


data['guardian'].unique()


# In[140]:


d = { 'mother':0,'father':1,'other':2}


# In[141]:


data['guardian'] = data['guardian'].map(d)


# In[142]:


data['address'].unique()


# In[143]:


d = {'U':0,'R':1}


# In[144]:


data['address'] = data['address'].map(d)


# In[145]:


data['famsize'].unique()


# In[146]:


d = {'GT3':0,'LE3':1}


# In[147]:


data['famsize'] = data['famsize'].map(d)


# In[148]:


data['Pstatus'].unique()


# In[149]:


d = {'A':0,'T':1}
   


# In[150]:


data['Pstatus'] = data['Pstatus'].map(d)


# In[151]:


from sklearn.model_selection import train_test_split


# In[152]:


x = data.drop('G3',axis=1)
y = data['G3']


# In[153]:


data['G3']


# In[154]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 ,random_state=42)


# In[155]:


from sklearn.linear_model import LinearRegression


# In[156]:


L = LinearRegression()


# In[157]:


L.fit(x_train, y_train)


# In[159]:


y_predict =  L.predict(x_test)


# In[160]:


print(L.score(x_test,y_test))


# 
