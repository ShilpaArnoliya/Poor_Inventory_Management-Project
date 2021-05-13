#!/usr/bin/env python
# coding: utf-8

# # Inventory Management
# 
# ## Business Objective:
# 
# Poor inventory management leads to a loss in sales which in turn paints an inaccurate picture of lower demand for certain items, making future order predictions based on that past data inherently inaccurate. Instead, smart retailers use real-time data to move inventory where it’s needed before it’s too late. Additionally, they use predictive analytics to decide what to stock and where based on data about regional differences in preferences, weather, etc
# 

# In[36]:


import warnings
warnings.filterwarnings('ignore')


# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


data1 = pd.read_csv('C:\\Users\\hp\\Downloads\\inventory_managment_project\\prorevenue.csv')
data2 = pd.read_csv('C:\\Users\\hp\\Downloads\\inventory_managment_project\\productdetails.csv')


# In[39]:


data1.head()


# In[40]:


data2.head()


# In[41]:


data2.drop(['Unnamed: 0'],inplace=True,axis=1)
data2.head()


# In[42]:


data1.rename(columns={'Product type':'product_type','No of purchases':'no_of_purchases','store status':'store-status','Promotion applied':'promotion_applied','Generic Holiday':'generic_holiday','Education Holiday':'education_holiday','DayOfWeek':'day_of_week'},inplace=True)
data1.head()


# In[43]:


data2.rename(columns={'product type':'product_type','cost per unit':'cost_per_unit','Time for delivery':'time_of_delivery'},inplace=True)
data2.head()


# In[44]:


data=pd.merge(data1,data2,on='product_type',how='right')
data.head(10)


# In[45]:


data.shape


# In[46]:


data.sample(10)


# In[47]:


data.columns


# No. of Purchases column is removed as it is irrelevant in the dataset

# In[48]:


data.drop(['no_of_purchases'],inplace=True,axis=1)
data.columns


# In[49]:


data.describe().transpose()


# In[50]:


data.dtypes


# In[51]:


data.nunique()


# In[52]:


data.isnull().sum()


# In[53]:


data.product_type.value_counts()


# In[54]:


data['generic_holiday'] = data['generic_holiday'].replace(['a','b','c'],['1','1','1']).astype(int)
data.generic_holiday.value_counts()


# In[55]:


data.duplicated(keep='first').sum()


# In[56]:


#Checking outliers
sns.boxplot(data=data,x=data["Revenue"])


# In[57]:


# Revenue Histogram
plt.hist(data.Revenue,bins=50, color='purple', edgecolor='black')
plt.title('Revenue')
plt.show()


# In[58]:


# Checking Outliers using IQR
Q1=data["Revenue"].quantile(0.25)
Q3=data["Revenue"].quantile(0.75)
IQR=Q3-Q1
print("Q1=",Q1)
print("Q3=",Q3)
print("IQR=",IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print("Lower whisker=",Lower_Whisker)
print("Upper Whisker=", Upper_Whisker)


# In[59]:


# Outlier Treatment
data = data[data["Revenue"]< Upper_Whisker]


# In[60]:


data.shape


# In[61]:


# Revenue data after removing outliers
sns.boxplot(data=data,x=data["Revenue"])


# In[62]:


# Sales = REVENUE/COST PER UNIT 
data["sales"] = round(data['Revenue']/data['cost_per_unit']) 
data.head(10)


# In[63]:


# Inventory = REVENUE/COST PER UNIT +10% Buffer stock
data['Inventory'] = round(data['sales'] + data['sales']* 0.1)


# In[64]:


data.head()


# In[65]:


data['store-status']= data['store-status'].replace(['open','close'],[1,0]).astype(int)
data.drop(['store-status'],inplace=True,axis=1)


# In[66]:


print(data[['sales', 'Inventory']])


# In[67]:


#data.drop(['sales'],inplace=True,axis=1)
data.columns


# In[68]:


data.corr()


# Revenue and store status are also highly correlated(0.75) followed by promotion applied(0.46). 

# In[69]:


print(data['Inventory'])


# In[70]:


weekly_data_Revenue = data.groupby('day_of_week').agg({'Revenue': ['min', 'max', 'sum','count', 'mean']})
print(weekly_data_Revenue)


# In[71]:


weekly_data_inventory = data.groupby('day_of_week').agg({'Inventory': ['min', 'max', 'sum','count', 'mean']})
print(weekly_data_inventory)


# In[72]:


data.head()


# ### Total columns deleted for further process are No. of purchases
# ### Target column is Inventory
# 

# In[73]:


new_data = data 


# ## Feature Selection

# In[74]:


new_data.head()


# In[75]:


# Dropping columns Revenue, Cost per unit and sales
new_data = new_data.drop(columns=['Revenue', 'cost_per_unit', 'promotion_applied','generic_holiday','education_holiday','day_of_week','time_of_delivery'])


# In[76]:


new_data.shape


# In[77]:


X = new_data.drop(columns=['Inventory'])
Y=new_data.Inventory


# In[78]:


X=X.astype('category')
X
# In[79]:


Y=Y.astype(int)
Y

# In[80]:


new_data.dtypes


# In[81]:


new_data['Inventory'].apply(np.ceil)


# In[82]:


X


# ### Robust Scaler

# In[83]:


from sklearn import preprocessing
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(X)
X = pd.DataFrame(robust_df)


# In[84]:


X.head()


# ## Model Building

# In[85]:


model_data=new_data.sample(frac=0.10)
model_data.head()


# In[86]:


model_data['Inventory'].apply(np.ceil)


# In[87]:


model_data.head()


# ## Random Forest Regressor

# In[88]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state = 42)


# In[89]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(X_train, y_train.values.ravel())


# In[90]:


y_pred = regressor.predict(X_test)
y_pred


# In[91]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[92]:


print("Train Accuracy:",regressor.score(X_train, y_train))
print("Test Accuracy:",regressor.score(X_test, y_test))


# In[93]:


import pickle
pickle_out = open("reg3.pkl", "wb")
pickle.dump(regressor,pickle_out)
pickle_out.close()


# In[94]:


# Predicting New data
print(regressor.predict([[1069, 4]]))


# In[ ]:




