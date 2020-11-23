#!/usr/bin/env python
# coding: utf-8

# In[33]:


#import labraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[34]:


df = pd.read_csv("D:\\Python Project\\level1\\wine\\/winequalityN.csv")


# In[35]:


df.head()


# In[36]:


df.describe()


# In[37]:


df.tail()


# In[38]:


df.info()


# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


df.isnull().sum()


# In[41]:


df["fixed acidity"].value_counts()


# In[42]:


mean = df["fixed acidity"].mean()
df["fixed acidity"].fillna(mean,inplace=True)
df["fixed acidity"].isnull().sum()


# In[43]:


mean2 = df["volatile acidity"].mean()
df["volatile acidity"].fillna(mean,inplace=True)
df["volatile acidity"].isnull().sum()


# In[44]:


df["citric acid"].value_counts()


# In[45]:


mean3 = df["citric acid"].mean()
df["citric acid"].fillna(mean,inplace=True)
df["citric acid"].isnull().sum()


# In[46]:


df["residual sugar"].value_counts()


# In[47]:


mean4 = df["residual sugar"].mean()
df["residual sugar"].fillna(mean,inplace=True)
df["residual sugar"].isnull().sum()


# In[48]:


mean4 = df["chlorides"].mean()
df["chlorides"].fillna(mean,inplace=True)
df["chlorides"].isnull().sum()


# In[49]:


mean5 = df["pH"].mean()
df["pH"].fillna(mean,inplace=True)
df["pH"].isnull().sum()


# In[50]:


mean6 = df["sulphates"].mean()
df["sulphates"].fillna(mean,inplace=True)
df["sulphates"].isnull().sum()


# In[51]:


df.isnull().sum()


# In[52]:


plt.figure(figsize=(10,7))
plt.scatter(x="alcohol",y="fixed acidity",data =df,marker= 'o',c="m")
plt.xlabel("alcohol",fontsize=15)
plt.ylabel("fixed_acidity",fontsize=15)
plt.show()


# In[53]:


sns.lmplot(x="alcohol",y="fixed acidity",data=df)
plt.plot()


# In[54]:


plt.figure(figsize=(10,7))
plt.scatter(x="volatile acidity",y="alcohol",data =df,marker= 'o',c="m")
plt.xlabel("volatile_acidity",fontsize=15)
plt.ylabel("alcohol",fontsize=15)
plt.show()


# In[55]:


sns.set(style="darkgrid")
sns.countplot(df["quality"],hue="type",data=df)
plt.show()


# In[56]:


sns.set()
sns.distplot(df["quality"],bins=10)
plt.show()


# In[57]:


plt.figure(figsize=(10,7))
sns.regplot(x="citric acid",y="chlorides",data =df,marker= 'o',color="m")
plt.show()


# In[58]:


sns.set()
sns.pairplot(df)
plt.show()


# In[59]:


sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data=df,palette="Set3")
plt.show()


# We can see that there are Some outliers.So now let's remove those Outliers

# In[60]:


lower_limit = df["free sulfur dioxide"].mean() - 3*df["free sulfur dioxide"].std()
upper_limit = df["free sulfur dioxide"].mean() + 3*df["free sulfur dioxide"].std()


# In[61]:


print(lower_limit,upper_limit)


# In[62]:


df2 = df[(df["free sulfur dioxide"] > lower_limit) & (df["free sulfur dioxide"] < upper_limit)]


# In[63]:


df.shape[0] - df2.shape[0]


# In[64]:


lower_limit = df2['total sulfur dioxide'].mean() - 3*df2['total sulfur dioxide'].std()
upper_limit = df2['total sulfur dioxide'].mean() + 3*df2['total sulfur dioxide'].std()
print(lower_limit,upper_limit)


# In[65]:


df3 = df2[(df2['total sulfur dioxide'] > lower_limit) & (df2['total sulfur dioxide'] < upper_limit)]
df3.head()


# In[66]:


df2.shape[0] - df3.shape[0]


# In[67]:


lower_limit = df3['residual sugar'].mean() - 3*df3['residual sugar'].std()
upper_limit = df3['residual sugar'].mean() + 3*df3['residual sugar'].std()
print(lower_limit,upper_limit)


# In[68]:


df4 = df3[(df3['residual sugar'] > lower_limit) & (df3['residual sugar'] < upper_limit)]
df4.head()


# In[69]:


df3.shape[0] - df4.shape[0]


# In[70]:


df4.isnull().sum()


# In[71]:


dummies = pd.get_dummies(df4["type"],drop_first=True)


# In[72]:


df4 = pd.concat([df4,dummies],axis=1)


# In[73]:


df4.drop("type",axis=1,inplace=True)


# In[75]:


df4.head()


# In[76]:


df4.quality.value_counts()


# Now lets Change the Categorical 'String' Variables into Numerical Variables

# In[77]:


quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
df4["quality"] =  df4["quality"].map(quaity_mapping)


# In[78]:


df4.quality.value_counts()


# In[79]:


df4.head()


# In[80]:


mapping_quality = {"Low" : 0,"Medium": 1,"High" : 2}
df4["quality"] =  df4["quality"].map(mapping_quality)


# In[81]:


df4.head()


# ### Machine Learning Model

# In[82]:


x = df4.drop("quality",axis=True)
y = df4["quality"]


# In[83]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)


# In[84]:


print(model.feature_importances_)


# In[85]:


feat_importances = pd.Series(model.feature_importances_,index =x.columns)
feat_importances.nlargest(9).plot(kind="barh")
plt.show()


# In[86]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[87]:


model_params  = {
    "svm" : {
        "model":SVC(gamma="auto"),
        "params":{
            'C' : [1,10,20],
            'kernel':["rbf"]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            'criterion':["entropy","gini"],
            "max_depth":[5,8,9]
        }
    },
    
    "random_forest":{
        "model": RandomForestClassifier(),
        "params":{
            "n_estimators":[1,5,10],
            "max_depth":[5,8,9]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params":{}
    },
    
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        'params': {
            "C" : [1,5,10]
        }
    }
    
}


# In[88]:


score=[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=8,return_train_score=False)
    clf.fit(x,y)
    score.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })


# In[89]:


df5 = pd.DataFrame(score,columns=["Model","Best_Score","Best_Params"])


# In[90]:


df5


# In[ ]:




