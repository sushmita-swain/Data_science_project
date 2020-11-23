#!/usr/bin/env python
# coding: utf-8

# In[115]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#for Spliting Data and Hyperparameter Tuning 
from sklearn.model_selection import train_test_split,GridSearchCV

#Importing Machine Learning Model
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

#statistical Tools
from sklearn import metrics

#To tranform data
from sklearn import preprocessing

#Setting Format
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.random.seed(100)


# In[116]:


# Importing Data

train = pd.read_csv("D:\\Python Project\\level1\\bigmart\\Train.csv")
test = pd.read_csv("D:\\Python Project\\level1\\bigmart\\Test.csv")


# In[117]:


data = pd.concat([train,test],ignore_index=True)
data.shape


# In[118]:


print(train.shape,test.shape)
print(train.shape[0]+test.shape[0])


# In[119]:


data.tail(10)


# ### Explorative Data Analysis

# In[120]:


data.info()


# In[121]:


data.describe(include='all').transpose()


# Lets Look at all the unique values in Columns

# In[122]:


#Extracting the unique values of each columns

for i in data.columns:
    print(i," : distinct_value")
    print(data[i].nunique(), ":No of unique values")
    print(data[i].unique())
    print("-"*30)
    print("")


# ### Univariate Analysis

# In[123]:


sns.boxplot(train.Item_Outlet_Sales)


# In[124]:


plt.hist(data.Item_Visibility, bins=10)
plt.show()


# In[125]:


plt.hist(data.Item_MRP, bins=10, rwidth=.8)
plt.show()


# In[126]:


plt.figure(figsize=(10,7))
for i, col in enumerate(['Item_Weight', 'Item_Visibility', 'Item_MRP']):
    plt.subplot(3,1,i+1)
    sns.boxplot(data[col])
    plt.xlabel('')
    plt.ylabel(col)


# In[127]:


plt.figure(figsize=(10,7))
for i, col in enumerate(['Item_Weight', 'Item_Visibility', 'Item_MRP']):
    plt.subplot(3,1,i+1)
    sns.violinplot(data[col])
    plt.xlabel('')
    plt.ylabel(col)


# In[128]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(data=data, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(data=data, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)
plt.show()


# In[129]:


plt.figure(figsize=(18,8))
sns.countplot(data.Item_Type)
plt.xticks(rotation=20)
plt.show()


# In[130]:


plt.figure(figsize=(18,8))
sns.boxplot(data=data, y='Item_Outlet_Sales', x='Item_Type')
plt.xlabel('Item_Type', fontsize=14)
plt.xticks(rotation=35)
plt.show()


# In[131]:


plt.figure(figsize=(18,8))
sns.countplot(data.Outlet_Identifier)
plt.xticks(rotation=20)
plt.show()


# In[132]:


plt.figure(figsize=(8,5))
sns.countplot(data.Outlet_Size)
plt.xticks(rotation=35)
plt.show()


# In[133]:


plt.figure(figsize=(18,8))
sns.countplot(data.Outlet_Establishment_Year)
plt.xticks(rotation=30)
plt.show()


# In[134]:


plt.figure(figsize=(12,8))
sns.countplot(data.Outlet_Type)
plt.xticks(rotation=45)
plt.show()


# In[135]:


plt.figure(figsize=(25,15))
outlet_col = [ 'Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type','Outlet_Establishment_Year']
for i, col in enumerate(outlet_col):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=data, y='Item_Outlet_Sales', x=col)
    plt.xlabel(col, fontsize=14)


# #### Multi-variate Analysis

# In[136]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
sns.scatterplot(data.Item_Weight, data.Item_Outlet_Sales, hue=data.Item_Type)


# In[137]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
sns.scatterplot(data.Item_Visibility, data.Item_Outlet_Sales, hue=data.Item_Type)


# ### Bivariate Analysis

# In[138]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Visibility")
plt.ylabel("Market Retail Price")
plt.title("Item_Visibility and Market Retail Price Analysis")
sns.scatterplot(data.Item_Visibility, data.Item_MRP, alpha=0.3)


# In[139]:


sns.scatterplot(data.Item_MRP, data.Item_Outlet_Sales)


# In[140]:


plt.figure(figsize=(18,8))
sns.boxplot(data.Item_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[141]:


plt.figure(figsize=(12,8))
sns.boxplot(data.Item_Fat_Content, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[142]:


plt.figure(figsize=(18,8))
sns.boxplot(data.Outlet_Identifier, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[143]:


sns.violinplot(data.Outlet_Location_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=20)
plt.show()


# In[144]:


plt.figure(figsize=(10,4))
sns.violinplot(data.Outlet_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=20)
plt.show()


# In[145]:


sns.pairplot(data.drop(columns='Outlet_Establishment_Year'), hue='Outlet_Type')


# ##### Finding Missing Values

# In[146]:


#We can see that we have some data missing

round(100*(data.isna().sum())/len(data), 2)


# In[147]:


#Looking for any '0' Values
data[data==0].sum()


# Filling the mssing data in Item_weight with mean
# 
# Because every item should have some weigh

# In[148]:


data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)


# In[149]:


data.head()


# #### Correlation

# In[150]:


data.corr()


# In[151]:


#Checking the Correlation between data with respect to target

data.corr()['Item_Outlet_Sales']


# In[152]:


plt.figure(figsize=(25,15))
sns.heatmap(data.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation between different attributes")


# In[153]:


data.head()


# ### Transform Data

# In[154]:


item_visiblity_avg = data.pivot_table( index = 'Item_Identifier', values = 'Item_Visibility')
item_visiblity_avg.head()


# In[155]:


data['Item_visiblity_avg'] = data.apply(lambda x: x['Item_Visibility']/item_visiblity_avg['Item_Visibility'][item_visiblity_avg.index==x['Item_Identifier']][0], axis=1).astype(float)


# In[156]:


data.head()


# In[157]:


data['Item_Visibility'], _ = stats.boxcox(data['Item_Visibility'] + 1)


# In[158]:


sns.distplot(data['Item_Visibility'])


# In[159]:


'''

We saw that in Item_identifier 1st 2 character's are common and other characters describes about the product

So we can extract those characters to simplify our dataset

'''

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'].value_counts()


# In[160]:


data.head()


# Rearranging the Columns To get the better view of data

# In[161]:


columns = ['Item_Identifier', 'Item_Type_Combined','Item_Type', 'Item_Fat_Content', 'Item_Weight', 'Item_Visibility',
        'Item_visiblity_avg', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']
data = data[columns]
data.head()


# In[162]:


data.pivot_table(values="Item_Outlet_Sales",index=['Item_Type_Combined','Item_Type'],aggfunc='sum')


# After we extract 1st 2 characters from Item_identifier we saw that it means to certain item type Now, we can rename them accordingly

# In[163]:


data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[164]:


data.Outlet_Size.value_counts()


# We saw that Outlet_size have some Nan Values Lets fill them with 'Not_specified' for now. We will work later for it.

# In[165]:


data['Outlet_Size'] = data['Outlet_Size'].fillna('Not_specified')
data['Outlet_Size'].value_counts()


# In[166]:


data['Item_Fat_Content'].value_counts()


# After looking closely we get that ["LF"," Low Fat"," low fat"] & ["Regular","reg"] are same and just acronyms

# In[167]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('LF','Low Fat')

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('low fat','Low Fat')

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg','Regular')

data['Item_Fat_Content'].value_counts()


# In[168]:


data.pivot_table(values="Item_Outlet_Sales",index=['Outlet_Location_Type','Outlet_Size','Outlet_Type'],aggfunc=np.sum)


# In[169]:


plot_outlet = data.pivot_table(values="Item_Outlet_Sales",index=['Outlet_Location_Type','Outlet_Size','Outlet_Type'],aggfunc=np.sum)
plot_outlet.plot(kind='bar',figsize = (10,6))
plt.xticks(rotation=75)
plt.show()


# In[170]:


plot_item = data.pivot_table(values="Item_Outlet_Sales",index=['Item_Fat_Content','Item_Type_Combined'],aggfunc='sum')
plot_item.plot(kind='bar',figsize = (10,6))
plt.xticks(rotation=0)
plt.show()


# In[171]:


data.pivot_table(values="Item_Outlet_Sales",index=['Item_Fat_Content','Item_Type_Combined','Item_Type'],aggfunc='sum')


# From above table we saw that Non_Consumable goods don't fit with Low_Fat. So lets change them accordingly Products looks like Household Items and lets it to Household

# In[172]:


data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Household"


# In[173]:


plot_identifier = data.pivot_table(index="Outlet_Identifier", values="Item_Outlet_Sales", aggfunc=np.sum)
plot_identifier


# In[174]:


plot_identifier.plot(kind='bar',figsize = (15,12))
plt.show()


# In[175]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Identifier'],aggfunc=np.sum)


# In[176]:


data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()


# In[177]:


data.Outlet_Identifier.value_counts()


# When we look Closely Grocery Store have the Lowest Sales
# 
# Supermarket Type3 have the Highest sale
# 
# After Carefully analyzing Outlet_size and Outlet_Identifier with Sales and Total Number of each Outlet_Identifier I think some Outlet_Identifier are in wrong Outlet_Size lets fix them

# In[178]:


data.loc[data['Outlet_Identifier']=="OUT010",'Outlet_Size'] = "Small"

data.loc[data['Outlet_Identifier']=="OUT019",'Outlet_Size'] = "Small"

data.loc[data['Outlet_Identifier']=="OUT027",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT017",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT045",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT035",'Outlet_Size'] = "Medium"

data.loc[data['Outlet_Identifier']=="OUT046",'Outlet_Size'] = "Medium"


# In[179]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Size','Outlet_Identifier'],aggfunc=np.sum)


# In[180]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Type','Outlet_Identifier'],aggfunc=np.sum).plot(kind='bar',figsize = (10,6))
plt.show()


# Let's Create New Columns

# In[181]:


perishable = ["Breads", "Breakfast", "Dairy", "Snack Foods",
               "Fruits and Vegetables", "Meat", "Seafood", "Starchy Foods"]

non_perishable = ["Baking Goods", "Canned", "Frozen Foods", 
                   "Hard Drinks", "Health and Hygiene",
                   "Household", "Soft Drinks"]


# In[182]:


def filter_data(item):
    if item in perishable:
        return 'perishable'
    elif item in non_perishable:
        return 'non_perishable'
    else:
        return'Not_Known'
    
data['Item_Type_New'] = data.Item_Type.apply(filter_data)


# In[183]:


data.Item_Type_New.value_counts()


# In[184]:


sns.scatterplot(data.Item_MRP, data.Item_Outlet_Sales)


# In[185]:


data['Item_MRP_cat'] = pd.cut(data.Item_MRP, bins=[31,69,137,203,270], labels=['a','b','c','d'])


# In[186]:


data.groupby(['Item_Type_Combined', 'Item_Type_New', 'Item_Type'])['Item_Outlet_Sales'].sum()


# In[187]:


data['MRP_per_unit_weight'] = data.Item_MRP/data.Item_Weight 


# In[188]:


data['Outlet_Years'] = 2020 - data['Outlet_Establishment_Year']


# In[189]:


data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('category')


# In[190]:


data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[3:5])


# In[191]:


data.head()


# In[192]:


data.corr()


# In[193]:


data['Item_visiblity_avg'] = np.log(data['Item_visiblity_avg'] + 1)


# In[194]:


plt.figure(figsize=(25,15))
sns.heatmap(data.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation between different attributes")


# In[195]:


df = data.copy()


# ### Seperating Train and test data

# In[196]:


df_train = df.iloc[0:train.shape[0]]
df_test = df.iloc[train.shape[0]:]


# In[197]:


print(df_train.shape,df_test.shape)


# In[198]:


df.info()


# In[199]:


df_train.select_dtypes(include='float').columns


# In[200]:


df_train = pd.get_dummies(df_train)


# In[201]:


df_train.corr()['Item_Outlet_Sales']


# In[202]:


df_train.head()


# In[203]:


df_train.shape


# In[204]:


df_train.info()


# In[205]:


df_x = df_train.drop(['Item_Outlet_Sales'], axis=1)
df_y = df_train['Item_Outlet_Sales']


# In[206]:


x_train, x_test, y_train, y_test=train_test_split(df_x,df_y, train_size=0.8, random_state=10)


# ### Machine Learning Model

# In[207]:


lr = LinearRegression()

rfc = ensemble.RandomForestRegressor(n_estimators=400, bootstrap=True, min_samples_leaf=100, min_samples_split=8, max_depth=6)
ada = ensemble.AdaBoostRegressor(n_estimators=1000, learning_rate=0.01)
gbr = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, min_samples_split=8, min_samples_leaf=100)
xgb = XGBRFRegressor(n_jobs=-1, n_estimators=1000, max_depth=5)
cat = CatBoostRegressor(verbose=0)
dtr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
lgbr = LGBMRegressor(n_estimators = 440, learning_rate=0.01, max_depth=12, objective='tweedie', num_leaves=15, num_threads = 4)

knn = KNeighborsRegressor()

mlp = MLPRegressor()

svr = SVR(kernel='linear', C=10, gamma='scale')


# In[208]:


accuracy = {}
rmse = {}
explained_variance = {}
max_error = {}
MAE = {}

def train_model(model, model_name):
    print(model_name)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    acc = metrics.r2_score(y_test, pred)*100
    accuracy[model_name] = acc
    print('R2_Score',acc)

    met = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('RMSE : ', met)
    rmse[model_name] = met

    var = (metrics.explained_variance_score(y_test, pred))
    print('Explained_Variance : ', var)
    explained_variance[model_name] = var

    error = (metrics.max_error(y_test, pred))
    print('Max_Error : ', error)
    max_error[model_name] = error
    
    err = metrics.mean_absolute_error(y_test, pred)
    print("Mean Absolute Error", err)
    MAE[model_name] = err


# In[209]:


train_model(cat, "Cat Boost")


# In[210]:


train_model(svr, "Support Vector Machine")


# In[211]:


train_model(lr, "Linear Regression")


# In[212]:


train_model(rfc, "Random Forest")


# In[213]:


train_model(xgb, "Xtreme Gradient Random Forest")


# In[214]:


train_model(ada, "Ada Boost")


# In[215]:


train_model(gbr, "Gradient Boost")


# In[216]:


train_model(dtr, "Decision Tree")


# In[217]:


train_model(mlp, "Multi-layer Perceptron")


# In[218]:


train_model(knn, "K Nearest Neighbors")


# In[219]:


train_model(lgbr, 'Light Gradient Boost')


# In[222]:


# from ngboost import NGBRegressor
# ngb = NGBRegressor(minibatch_frac=0.5, col_sample=0.5, Base=dtr)
# train_model(ngb, "Natural Grading Boost")


# ### Visualising our Output

# In[223]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,110,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette="cubehelix")
plt.show()


# In[224]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1600,100))
plt.ylabel("MSE")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(rmse.keys()), y=list(rmse.values()))
plt.show()


# In[225]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1,0.05))
plt.title("Explained Variance Score")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(explained_variance.keys()), y=list(explained_variance.values()))
plt.show()


# In[226]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,10000,500))
plt.ylabel("Max Error")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(max_error.keys()), y=list(max_error.values()))
plt.show()


# In[241]:


sns.set_style("whitegrid")
plt.figure(figsize=(16, 5))
plt.yticks(np.arange(0, 1000, 100))
plt.ylabel("Mean Absolute Error")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x = list(MAE.keys()), y = list(MAE.values()))
plt.show()


# In[ ]:




