#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pickle
import pandas as pd
df=pd.read_csv(r'C:\Users\Computer\Desktop\project\dataset\carrot.csv')
print(df)


# In[135]:


df.drop('crop', inplace=True, axis=1)
print(df)


# In[136]:


df = df.dropna()


# In[137]:


from sklearn.ensemble import IsolationForest
forest = IsolationForest(random_state=0)
forest.fit(df)


# In[67]:


scores = forest.score_samples(df)
print(scores)


# In[115]:


comp=1
ans=-1
for i in range (0,1122):
    if(scores[i]<comp):
        comp=scores[i]
        ans=i
print(ans)


# In[127]:



file = open(r'C:\Users\Computer\Downloads\carrot.csv','wb')

pickle.dump(forest,file)


# In[128]:


file.fit(df)

