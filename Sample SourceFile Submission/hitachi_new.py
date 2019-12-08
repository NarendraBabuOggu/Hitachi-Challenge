#!/usr/bin/env python
# coding: utf-8

# # TRAIN PASSENGER DEMAND PREDICTION

# ### Importing the required libriaries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data

# In[2]:


cwd = r"D:\Narendra\Hitachi data Engineer Challenge\DataSet"
train_data = pd.read_csv(cwd + "/train.csv")
test_data = pd.read_csv(cwd + "/test.csv")
columns=train_data.drop(['id_code', 'target'], axis=1).columns
tot_data = pd.concat([train_data[columns], test_data[columns]])


# ## Data Cleansing and Feature Engineering 

# ### Creating a Datetime column from the data

# In[3]:


train_data['current_datetime'] = pd.to_datetime(train_data['current_date']+' '+train_data['current_time'])
train_data['datetime'] = pd.to_datetime(train_data['current_date']+' '+train_data['current_time'])
test_data['current_datetime'] = pd.to_datetime(test_data['current_date']+' '+test_data['current_time'])
test_data['datetime'] = pd.to_datetime(test_data['current_date']+' '+test_data['current_time'])
test_data.set_index('current_datetime', inplace=True)
train_columns = train_data.columns


# ## Creating the Data Model with Station, Country, date Dimensions and Train details as Fact 

# ### Station Dimension

# In[4]:


source_station = tot_data[['source_name', 'country_code_source', 'longitude_source', 'latitude_source', 'mean_halt_times_source']]
destination_station = tot_data[['destination_name', 'country_code_destination', 'longitude_destination', 'latitude_destination', 'mean_halt_times_destination']]
source_station.columns=['name', 'country_code', 'latitude', 'longitude', 'mean_halt_times']
destination_station.columns=['name', 'country_code', 'latitude', 'longitude', 'mean_halt_times']


# In[5]:


station_detail = pd.concat([source_station, destination_station])
station_detail['id'] = station_detail['name'].apply(lambda x: int(x[8:]))
station_detail = station_detail.drop_duplicates()
station_detail.sort_values('id', inplace=True)
station_detail.reset_index(drop=True, inplace=True)
station_detail = station_detail.fillna(method = 'bfill') #Filling NaN values with nearby sation values
station_detail


# In[6]:


station_dict = {}
for x,y in zip(station_detail['name'], station_detail['id']):
    station_dict[x] = y


# ### Country Dimension 

# In[7]:


country_code = station_detail['country_code'].sort_values()
country_code = country_code.drop_duplicates()
country_code.reset_index(drop=True, inplace=True)
country_code = country_code.reset_index(drop=False)
country_code.columns=['id', 'code']
country_code


# In[8]:


country_dict={}
for x,y in zip(country_code['code'], country_code['id']):
    country_dict[x]=y


# ### Train details Fact 

# In[9]:


train_detail = train_data.loc[:,['id_code', 'current_datetime', 'train_name', 'source_name', 'destination_name', 'target']]
train_detail = train_detail.drop_duplicates()
train_detail.sort_values('current_datetime', inplace=True)
train_detail.reset_index(drop=True, inplace=True)
train_detail.reset_index(inplace=True)
train_detail.rename(columns={'current_datetime':'datetime', 'index':'train_id', 'target':'passenger_demand', 
                             'source_name':'source', 'destination_name': 'destination'}, inplace=True)
train_detail


# In[10]:


train_dict = {}
for x,y in zip(train_detail['train_name'], train_detail['train_id']):
    train_dict[x] = y


# ### Datetime Dimension 

# In[11]:


datetime_detail = pd.DataFrame(pd.date_range('2016-06-01', '2016-11-30', freq='s'), columns=['datetime'])
datetime_detail['datetime'] = pd.to_datetime(datetime_detail['datetime'])
datetime_detail['day'] = datetime_detail['datetime'].dt.weekday.astype(int)
datetime_detail['is_weekend']=0
datetime_detail.loc[datetime_detail['day']>4, 'is_weekend']=1
datetime_detail['hours'] = datetime_detail['datetime'].dt.hour.astype(int)
datetime_detail['minutes'] = datetime_detail['datetime'].dt.minute.astype(int)
datetime_detail.reset_index(inplace=True)
datetime_detail.rename(columns={'index':'id'}, inplace=True)


# In[12]:


def get_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# In[13]:


columns_with_nulls = train_data.columns[train_data.isna().sum()>0]
columns_with_nulls


# ### Updating the missing values in train_data and calculating the Distance between source and destination stations 

# In[14]:


train_data = train_data.merge(station_detail, left_on='source_name', right_on='name', how='left')

train_data = train_data.merge(station_detail, left_on='destination_name', right_on='name', how='left', suffixes=('_source', '_destination'))

train_data = train_data.dropna(axis=1)[train_columns]
train_data.set_index('current_datetime', inplace=True)

train_data['distance_between_stations'] = get_distance(train_data['latitude_source'].values, train_data['longitude_source'].values, 
                                                       train_data['latitude_destination'].values, train_data['longitude_destination'].values)


# In[15]:


# checking null values
train_data.isna().sum()


# In[16]:


#Separating the target values for analysis
train_data['passenger_demand_high'] = np.where(train_data['target']=='high', 1, 0)
train_data['passenger_demand_medium'] = np.where(train_data['target']=='medium', 1, 0)
train_data['passenger_demand_low'] = np.where(train_data['target']=='low', 1, 0)


# ### Extending the datetime features

# In[17]:


train_data['hours'] = train_data['datetime'].dt.hour.astype(int)
train_data['minutes'] = train_data['datetime'].dt.minute.astype(int)
train_data['sec'] = train_data['datetime'].dt.second.astype(int)
train_data['is_weekend'] = (train_data['is_weekend']*1).astype(int)
train_data['current_day'] = train_data['datetime'].dt.weekday.astype(int)


# ### Labeling the Target into numeric values

# ### Plotting trains with high, medium and low separately 

# In[18]:


x_values = train_data.loc[train_data['target']=='high', 'datetime']
y_values = train_data.loc[train_data['target']=='high', 'id_code']
plt.figure(figsize=(20,5))
plt.scatter(x_values, y_values)
x_values = train_data.loc[train_data['target']=='medium', 'datetime']
y_values = train_data.loc[train_data['target']=='medium', 'id_code']
plt.figure(figsize=(20,5))
plt.scatter(x_values, y_values)
x_values = train_data.loc[train_data['target']=='low', 'datetime']
y_values = train_data.loc[train_data['target']=='low', 'id_code']
plt.figure(figsize=(20,5))
plt.scatter(x_values, y_values)


# In[19]:


train_data['target'] = np.where(train_data['target']=='low', 0, np.where(train_data['target']=='medium', 1, 2))


# ### Plotting the data to see the trends with respect to time(it is clear that the relationship is varying)

# In[20]:


train_data.groupby('datetime')[['passenger_demand_high', 'passenger_demand_medium', 'passenger_demand_low']].mean().plot(figsize=(20,5))


# ### Calculating the probabilities of each train to be high, medium and low in demand

# In[21]:


trains_df = train_data.groupby(['train_name'])['passenger_demand_high', 'passenger_demand_medium', 'passenger_demand_low'].mean()
trains_df.columns = ['train_high_prob', 'train_medium_prob', 'train_low_prob']


# ### Adding the train probabilities into training data

# In[22]:


trains_df.reset_index(inplace=True)
train_data = train_data.merge(trains_df, left_on='train_name', right_on = 'train_name', how='left')


# ### Calculating the probabilities of each source and destination station to be high, medium and low in demand

# In[23]:


stations_df = train_data.groupby(['source_name', 'destination_name'])['passenger_demand_high', 'passenger_demand_medium', 'passenger_demand_low'].mean()
stations_df.columns = ['station_high_prob', 'station_medium_prob', 'station_low_prob']


# ### Adding the station probability data into training data 

# In[24]:


stations_df.reset_index(inplace=True)
train_data = train_data.merge(stations_df, left_on=['source_name', 'destination_name'], right_on = ['source_name', 'destination_name'], how='left')


# ### Calculating the probabilities of each day to be high, medium and low in demand

# In[25]:


days_df = train_data.groupby(['current_day'])['passenger_demand_high', 'passenger_demand_medium', 'passenger_demand_low'].mean()
days_df.columns = ['day_high_prob', 'day_medium_prob', 'day_low_prob']


# ### Adding the day probability data into training data 

# In[26]:


days_df.reset_index(inplace=True)
train_data = train_data.merge(days_df, left_on='current_day', right_on = 'current_day', how='left')


# ### Calculating the probabilities of each hour to be high, medium and low in demand

# In[27]:


hours_df = train_data.groupby(['hours'])['passenger_demand_high', 'passenger_demand_medium', 'passenger_demand_low'].mean()
hours_df.columns = ['hour_high_prob', 'hour_medium_prob', 'hour_low_prob']


# ### Adding the hour probability data into training data 

# In[28]:


hours_df.reset_index(inplace=True)
train_data = train_data.merge(hours_df, left_on='hours', right_on = 'hours', how='left')


# ### Analysing the trends in source and destination country

# ### Transformong and Adding extra features to test data

# In[29]:


test_data['current_day'] = test_data['datetime'].dt.weekday.astype(int)
test_data['hours'] = test_data['datetime'].dt.hour.astype(int)
test_data['minutes'] = test_data['datetime'].dt.minute.astype(int)
test_data['sec'] = test_data['datetime'].dt.second.astype(int)
test_data['is_weekend'] = (test_data['is_weekend']*1).astype(int)
test_data['distance_between_stations'] = get_distance(test_data['latitude_source'].values, test_data['longitude_source'].values, 
                                                       test_data['latitude_destination'].values, test_data['longitude_destination'].values)


# In[30]:


test_data = test_data.merge(trains_df, left_on='train_name', right_on='train_name', how='left')
test_data = test_data.merge(stations_df, left_on=['source_name', 'destination_name'], right_on=['source_name', 'destination_name'], how='left')
test_data = test_data.merge(days_df, left_on='current_day', right_on = 'current_day', how='left')
test_data = test_data.merge(hours_df, left_on='hours', right_on = 'hours', how='left')


# ### Updating the missing probabilities in testing data with default probability (1/3) 

# In[31]:


test_na_cols = test_data.columns[test_data.isna().sum()>0]
test_data[test_na_cols] = test_data[test_na_cols].fillna(1/3)


# ### Scaling the numerical feature columns to fit the model 

# In[32]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler().fit(train_data[['mean_halt_times_source', 'mean_halt_times_destination', 'distance_between_stations']])
train_data.loc[:,['mean_halt_times_source', 'mean_halt_times_destination', 'distance_between_stations']] = scale.transform(train_data[['mean_halt_times_source', 'mean_halt_times_destination', 'distance_between_stations']])
test_data.loc[:,['mean_halt_times_source', 'mean_halt_times_destination', 'distance_between_stations']] = scale.transform(test_data[['mean_halt_times_source', 'mean_halt_times_destination', 'distance_between_stations']])                             


# In[33]:


station_detail['country_code'] = station_detail['country_code'].map(country_dict)
train_detail['source'] = train_detail['source'].map(station_dict)
train_detail['destination'] = train_detail['destination'].map(station_dict)


# ### Preparing the datasets for train and validation

# In[34]:


train_columns = ['train_high_prob', 'train_low_prob', 'train_medium_prob', 'station_high_prob', 'station_medium_prob', 
                 'station_low_prob', 'hour_high_prob', 'hour_medium_prob', 'day_high_prob', 'day_medium_prob', 'day_low_prob', 
                 'hour_low_prob', 'mean_halt_times_source', 'mean_halt_times_destination', 'is_weekend', 'distance_between_stations'] 
X = train_data[train_columns]
Y = train_data['target']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, stratify=Y)


# ### Analysing with Random Forest Classifier Model

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
print(rf_clf.score(x_train, y_train))
print(rf_clf.score(x_test, y_test))


# In[36]:


print(confusion_matrix(rf_clf.predict(x_test), y_test))
print(f1_score(rf_clf.predict(x_test), y_test, average='macro'))


# In[37]:


submission = {}
submission['id_code'] = test_data['id_code']
test_pred = rf_clf.predict(test_data[train_columns])
submission['target']=np.where(test_pred==0, 'low', np.where(test_pred==1, 'medium', 'high'))
submission=pd.DataFrame(submission)
print(np.unique(submission['target'], return_counts=True))
submission.to_csv(cwd + "/submission_v1.csv", index=False)


# ### Analysing with Extra Trees Classifier Model 

# In[38]:


from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(criterion='entropy', n_estimators=30, bootstrap=True, max_features=None, max_depth=7, max_leaf_nodes=7)
et_clf


# In[39]:


et_clf.fit(x_train, y_train)
print(et_clf.score(x_train, y_train))
print(et_clf.score(x_test, y_test))


# In[40]:


print(confusion_matrix(et_clf.predict(x_test), y_test))
print(f1_score(et_clf.predict(x_test), y_test, average='macro'))


# In[41]:


test_pred = et_clf.predict(test_data[train_columns])
submission['target']=np.where(test_pred==0, 'low', np.where(test_pred==1, 'medium', 'high'))
print(np.unique(submission['target'], return_counts=True))
submission.to_csv(cwd + "/submission_v2.csv", index=False)


# ### Analysing with Logistic Regression Model

# In[42]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=0.01, multi_class='multinomial', solver='lbfgs', max_iter=100)
log_reg


# In[43]:


log_reg.fit(x_train, y_train)
print(log_reg.score(X, Y))
print(log_reg.score(x_test, y_test))
print(confusion_matrix(log_reg.predict(x_test), y_test))
print(f1_score(log_reg.predict(x_test), y_test, average='macro'))


# In[44]:


#Only Logistic Regression
test_pred = log_reg.predict(test_data[train_columns])
submission['target']=np.where(test_pred==0, 'low', np.where(test_pred==1, 'medium', 'high'))
print(np.unique(submission['target'], return_counts=True))
submission.to_csv(cwd + "/submission_v3.csv", index=False)


# ### Analysing with LGBM (Light Gradient Boosting Model) 

# In[45]:


train_columns = ['train_high_prob', 'train_low_prob', 'train_medium_prob', 'station_high_prob', 'station_medium_prob', 
                 'station_low_prob'] 
X = train_data[train_columns]
Y = train_data['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y)


# In[46]:


import lightgbm as lgb
params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':3,
            'metric': {'multi_logloss'},
            'subsample': 0.7,
            'learning_rate': 0.04,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'pos_bagging_fraction':0.7,
            'neg_bagging_fraction':0.3,
            'bagging_freq':10,
            'is_unbalance':True,
            'num_leaves': 31,
            'max_depth': 5,
            'random_state':3
            }

lgb_train = lgb.Dataset(X, Y)
lgb_eval = lgb.Dataset(x_test, y_test)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=100,
               verbose_eval = 100)


# In[47]:


from sklearn.metrics import accuracy_score
print(confusion_matrix(np.argmax(gbm.predict(x_test), axis=1), y_test))
print("F1 Score : ",f1_score(np.argmax(gbm.predict(x_test), axis=1), y_test, average='macro'))
print("Accuracy Score : ", accuracy_score(np.argmax(gbm.predict(x_test), axis=1), y_test))


# In[48]:


test_pred = np.argmax(gbm.predict(test_data[train_columns]), axis=1)
submission['target']=np.where(test_pred==0, 'low', np.where(test_pred==1, 'medium', 'high'))
print(np.unique(submission['target'], return_counts=True))
submission.to_csv(cwd + "/submission_v4.csv", index=False)


# In[49]:


train_data.to_csv(r"D:\Narendra\Hitachi data Engineer Challenge\DataSet\Sample SourceFile Submission/re-train.csv", index=False)


# In[ ]:




