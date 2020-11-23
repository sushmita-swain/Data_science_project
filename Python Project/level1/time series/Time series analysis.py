#!/usr/bin/env python
# coding: utf-8

# Time Series Analysis and forecasting using ARIMA
# 
# 
# What is a time series problem
# 
# 
# In the field for machine learning and data science, most of the real-life problems are based upon the prediction of future which is totally oblivious to us such as stock market prediction, future sales prediction and so on.Time series problem is basically the prediction of such problems using various machine learning tools.Time series problem is tackled efficiently when first it is analyzed properly (Time Series Analysis) and according to that observation suitable algorithm is used (Time Series Forecasting).We'll study both of then later in this notebook.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# In[2]:


# Reading the data
df = pd.read_csv("D:\\Python Project\\level1\\portland-oregon-average-monthly-.csv")


# In[3]:


# A glance on the data 
df.head()


# In[4]:


# getting some information about dataset
df.info()


# From this I can infer two necessary things:
# 
# I really need to change change columns name
# 
# 
# Both the columns have object datatype

# In[6]:


# further Analysis 
df.describe()


# In[7]:


df.columns = ["month", "average_monthly_ridership"]
df.head()


# In[8]:


df.dtypes


# In[9]:


df['average_monthly_ridership'].unique()


# We can see here that this series consist an anamolous data which is the last one.

# In[10]:


df = df.drop(df.index[df['average_monthly_ridership'] == ' n=114'])


# In[11]:


df['average_monthly_ridership'].unique()


# Now our data is clean !!!
# 
# Changing data type of both the column
# 
# Assign int to monthly_ridership_data column
# 
# Assign datetime to month column

# In[12]:


df['average_monthly_ridership'] = df['average_monthly_ridership'].astype(np.int32)


# In[13]:


df['month'] = pd.to_datetime(df['month'], format = '%Y-%m')


# In[14]:


df.dtypes


# Time Series Analysis
# 
# As you all know how important data analysis is for data scientists.It gives us a brief understanding of the data and a very strange but intriguing confidence about our prediction model.Well, Time series analysis is no different.But time series problems have very special orientation when it comes to analysis.But before we move into that, let me introduce you to some jargons (Just Kidding it is pure and simple english) which are frequently used in this problem domain.
# 
# Trend:- As the name suggests trend depicts the variation in the output as time increases.It is often non-linear. Sometimes we will refer to trend as “changing direction” when it might go from an increasing trend to a decreasing trend.
# 
# Level:- It basically depicts baseline value for the time series.
# 
# Seasonal:- As its name depicts it shows the repeated pattern over time. In layman terms, it shows the seasonal variation of data over time.
# 
# Noise:- It is basically external noises that vary the data randomly.
# 
# 

# In[15]:


# Normal line plot so that we can see data variation
# We can observe that average number of riders is increasing most of the time
# We'll later see decomposed analysis of that curve
df.plot.line(x = 'month', y = 'average_monthly_ridership')
plt.show()


# Ploting monthly variation of dataset

# In[16]:


to_plot_monthly_variation = df


# In[17]:


# only storing month for each index 
mon = df['month']


# In[18]:


# decompose yyyy-mm data-type 
temp= pd.DatetimeIndex(mon)


# In[19]:


# assign month part of that data to ```month``` variable
month = pd.Series(temp.month)


# In[20]:


# dropping month from to_plot_monthly_variation
to_plot_monthly_variation = to_plot_monthly_variation.drop(['month'], axis = 1)


# In[21]:


# join months so we can get month to average monthly rider mapping
to_plot_monthly_variation = to_plot_monthly_variation.join(month)


# In[22]:


# A quick glance
to_plot_monthly_variation.head()


# In[23]:


# Plotting bar plot for each month
sns.barplot(x = 'month', y = 'average_monthly_ridership', data = to_plot_monthly_variation)
plt.show()


# Well this looks tough to decode. Not a typical box plot. One can infer that data is too sparse for this graph to represent any pattern. Hence it cannot represents monthly variation effectively.In such a scenerio we can use our traditional scatter plot to understand pattern in dataset

# In[24]:


to_plot_monthly_variation.plot.scatter(x = 'month', y = 'average_monthly_ridership')
plt.show()


# We can see here the yearly variation of data in this plot. To understand this curve more effectively first look at the every row from bottom to top and see each year's variation.To understand yearly variation take a look at each column representing a month.
# 
# Another tool to visualize the data is the seasonal_decompose function in statsmodel. With this, the trend and seasonality become even more obviou

# In[25]:


rider = df[['average_monthly_ridership']]


# ### Trend Analysis

# In[26]:


rider.rolling(6).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()


# For trend analysis, we use smoothing techniques. In statistics smoothing a data set means to create an approximating function that attempts to capture important patterns in the data, while leaving out noise or other fine-scale structures/rapid phenomena. In smoothing, the data points of a signal are modified so individual points (presumably because of noise) are reduced, and points that are lower than the adjacent points are increased leading to a smoother signal.We implement smoothing by taking moving averages. Exponential moving average is frequently used to compute smoothed function.Here I used the rolling method which is inbuilt in pandas and frequently used for smoothing.

# ### Seasonability Analysis

# In[27]:


rider.diff(periods=4).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()


# The above figure represents difference between average rider of a month and 4 months before that month i.e
# 
# d[month]=a[month]−a[month−periods] .
# 
# This gives us idea about variation of data for a period of time.

# Periodicity and Autocorrelation

# In[28]:


pd.plotting.autocorrelation_plot(df['average_monthly_ridership'])
plt.show()


# In[29]:


pd.plotting.lag_plot(df['average_monthly_ridership'])
plt.show()


# The above curve represents the relation between current time stepp and its previous time step

# In[30]:


df = df.set_index('month')


# In[31]:


# Applying Seasonal ARIMA model to forcast the data 
mod = sm.tsa.SARIMAX(df['average_monthly_ridership'], trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()
print(results.summary())


# In[32]:


df['forecast'] = results.predict(start = 102, end= 120, dynamic= True)  
df[['average_monthly_ridership', 'forecast']].plot(figsize=(12, 8))
plt.show()


# In[33]:


def forcasting_future_months(df, no_of_months):
    df_perdict = df.reset_index()
    mon = df_perdict['month']
    mon = mon + pd.DateOffset(months = no_of_months)
    future_dates = mon[-no_of_months -1:]
    df_perdict = df_perdict.set_index('month')
    future = pd.DataFrame(index=future_dates, columns= df_perdict.columns)
    df_perdict = pd.concat([df_perdict, future])
    df_perdict['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  
    df_perdict[['average_monthly_ridership', 'forecast']].iloc[-no_of_months - 12:].plot(figsize=(12, 8))
    plt.show()
    return df_perdict[-no_of_months:]


# In[34]:


predicted = forcasting_future_months(df,10)


# In[35]:


df.tail()


# In[ ]:




