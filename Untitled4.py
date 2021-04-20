#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import xlrd
import seaborn as sns
from IPython.core.debugger import set_trace
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[104]:



df = pd.read_csv (r'C:\Users\Computer\Desktop\project\dataset\carrot.csv')    
num=df.shape[0]
print(num)
df.head(2)


# In[ ]:





# In[114]:


contamination = 0.001
data=df.copy()


# In[115]:



del data["crop"]
data = data.dropna()
model = IsolationForest(contamination=contamination, n_estimators=1000)
model.fit(data)


# In[116]:


df["iforest"] = pd.Series(model.predict(data))
df["iforest"] = df["iforest"].map({1: 0, -1: 1})
print(df["iforest"].value_counts())


# In[117]:


count=0
for i in range (0,num):
    #if(ans[i]==1):
    if(df["iforest"][i]==1):
        print(i)
#print(i)


# In[113]:


a=df["iforest"][1118]
print(a)

