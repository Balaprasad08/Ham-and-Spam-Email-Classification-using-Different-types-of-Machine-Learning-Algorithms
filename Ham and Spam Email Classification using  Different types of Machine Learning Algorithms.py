#!/usr/bin/env python
# coding: utf-8

# ### Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\NLP\\Ham and Spam Email Classification using  Different types of Machine Learning Algorithms')


# ### Perform Imports and Load Data

# In[3]:


df=pd.read_table('smsspamcollection.tsv',sep='\t')


# In[4]:


df.head(2)


# In[5]:


df.shape


# In[6]:


len(df)


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.info()


# ### Visualize the data:

# In[10]:


df.label.value_counts()


# In[11]:


plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label']=='ham']['length'],bins=bins,label='Ham')
plt.hist(df[df['label']=='spam']['length'],bins=bins,label='Spam')
plt.legend()
plt.show()


# In[12]:


sns.countplot(df.label)
plt.show()


# In[13]:


plt.xscale('log')
bins = 1.5**(np.arange(0,15))
plt.hist(df[df['label']=='ham']['punct'],bins=bins,label='Ham')
plt.hist(df[df['label']=='spam']['punct'],bins=bins,label='Spam')
plt.legend()
plt.show()


# ### Create Feature and Label sets

# In[14]:


X=df[['length','punct']]
X.head(2)


# In[15]:


y=df['label']
y.head(2)


# ### Data Split into Train,Test

# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Model Building

# In[17]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[18]:


lr=LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
accuracy_score(y_test,y_pred)


# In[19]:


cm=confusion_matrix(y_test,y_pred)
cm


# #### Create Function For Model Evaluation

# In[20]:


def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('Accuracy Score: ',accuracy_score(y_test,y_pred))
    print('\n','Confusion Matrix:')
    print(confusion_matrix(y_test,y_pred))
    print('\n','Classification Report:','\n')
    print(classification_report(y_test,y_pred))


# #### Check Accuracy_score by using different algorithms

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC


# In[22]:


# LogisticRegression
check_model(LogisticRegression(),X_train,X_test,y_train,y_test)


# In[23]:


# RandomForestClassifier
check_model(RandomForestClassifier(),X_train,X_test,y_train,y_test)


# In[24]:


# KNeighborsClassifier
check_model(KNeighborsClassifier(),X_train,X_test,y_train,y_test)


# In[25]:


# DecisionTreeClassifier
check_model(DecisionTreeClassifier(),X_train,X_test,y_train,y_test)


# In[26]:


# MultinomialNB
check_model(MultinomialNB(),X_train,X_test,y_train,y_test)


# In[27]:


# GaussianNB
check_model(GaussianNB(),X_train,X_test,y_train,y_test)


# In[28]:


# SVC
check_model(SVC(),X_train,X_test,y_train,y_test)


# In[ ]:




