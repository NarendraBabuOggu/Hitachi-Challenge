# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:28:38 2019

@author: 91998
"""

import numpy as np
import pandas as pd
import holidays

holiday_list = holidays.India(years=2016)

train_data = pd.read_csv(r"D:\Narendra\Hitachi data Engineer Challenge\DataSet/train.csv")
test_data = pd.read_csv(r"D:\Narendra\Hitachi data Engineer Challenge\DataSet/test.csv")

print(train_data.shape, test_data.shape)
print(train_data.columns, test_data.columns)
train_data['current_date'] = pd.to_datetime(train_data['current_date'])
train_data['current_time'] = pd.to_datetime(train_data['current_time'])
train_data['hour'] = train_data['current_time'].dt.hour
train_data['current_month'] = train_data['current_date'].dt.month.astype(int)

test_data['current_date'] = pd.to_datetime(test_data['current_date'])
test_data['current_time'] = pd.to_datetime(test_data['current_time'])
test_data['hour'] = test_data['current_time'].dt.hour
test_data['current_month'] = test_data['current_date'].dt.month.astype(int)

#weather_train_csv['Season'] = 
def get_season(month, day):
    if month in (1, 2, 3):
        season = 1
    elif month in (4, 5, 6):
        season = 2
    elif month in (7, 8, 9):
        season = 3
    else:
        season = 4

    if (month == 3) and (day > 19):
        season = 2
    elif (month == 6) and (day > 20):
        season = 3
    elif (month == 7) and (day > 21):
        season = 4
    elif (month == 12) and (day > 20):
        season = 1
    return season

season = []
for day, month in zip(train_data["current_month"], train_data["current_day"]):
    season.append(get_season(month, day))
train_data['season'] = season
season = []
for day, month in zip(test_data["current_month"], test_data["current_day"]):
    season.append(get_season(month, day))
test_data['season'] = season

def get_holidays(dates):
    is_holiday=[]
    for date in dates:
        if date in holiday_list:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    return np.array(is_holiday)

train_data['is_holiday'] = get_holidays(train_data['current_date'])
test_data['is_holiday'] = get_holidays(test_data['current_date'])


columns = train_data.columns
train_columns = columns.drop(['id_code', 'target', 'current_date', 'current_time', 'current_year'])
#print(train_data.nunique())
categoricals = ['source_name', 'destination_name', 'train_name', 'country_code_source', 'country_code_destination', 'current_week', 
                'current_day', 'is_weekend', 'season', 'hour', 'current_month', 'is_holiday']

train_data['country_code_source'] = train_data['country_code_source'].fillna('unk')
train_data['country_code_destination'] = train_data['country_code_destination'].fillna('unk')

test_data['country_code_source'] = test_data['country_code_source'].fillna('unk')
test_data['country_code_destination'] = test_data['country_code_destination'].fillna('unk')

tot_data = pd.concat([train_data[train_columns], test_data[train_columns]], axis=0)

unique_df = tot_data[categoricals].nunique().reset_index()
unique_df.columns = ['column', 'unique_count']

def to_categorical(data, unique):
    new_data = []
    for i in range(len(unique)):
        new_data.append(np.where(data==unique[i],1,0))
    return np.array(new_data).T

unique_values = {}
for col in categoricals:
    unique = np.unique(tot_data[col])
    unique_values[col]=unique


for col in categoricals:
    if col in ['is_weekend', 'is_holiday']:
        continue
    else:
        print(col)
        count = len(unique_values[col])
        col_names = [col+str(i) for i in range(count)]
        train_data[col_names] = pd.DataFrame(to_categorical(train_data[col], unique_values[col]), columns=col_names)

for col in categoricals:
    if col in ['is_weekend', 'is_holiday']:
        continue
    else:
        print(col)
        count = len(unique_values[col])
        col_names = [col+str(i) for i in range(count)]
        test_data[col_names] = pd.DataFrame(to_categorical(test_data[col], unique_values[col]), columns=col_names)

unique_values['target'] = np.unique(train_data['target'])
train_data[unique_values['target']] = pd.DataFrame(to_categorical(train_data['target'], unique_values['target']), columns=unique_values['target'])

train_data = train_data.dropna(axis=0, how='any')
sparse_columns = [col for col in train_columns if col not in categoricals]
train_data = train_data.drop(categoricals, axis=1)
columns = train_data.columns
train_columns = columns.drop(['id_code', 'target', 'high', 'low', 'medium', 'current_date', 'current_time', 'current_year'])

X = train_data[train_columns]
Y = train_data[unique_values['target']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y)

from sklearn.neighbors import KNeighborsClassifier
for i in range(1, 10):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train)   
    print(clf.score(x_train, y_train), clf.score(x_test, y_test))
    
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1)
clf.fit(x_train, y_train)   
print(clf.score(x_train, y_train), clf.score(x_test, y_test))
imp_feat = clf.feature_importances_
for col, imp in zip(train_columns, imp_feat):
    if imp>0.01:
        print(col, imp)

X = train_data[train_columns]
Y = np.where(train_data['target']=='high', 1, (np.where(train_data['target']=='medium', 2, 3))) 
Y = Y.astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y)

from sklearn.svm import SVC
for kernel_ in ['linear', 'poly', 'rbf']:
    clf = SVC(kernel=kernel_, gamma='auto', decision_function_shape='ovo', degree=4)
    clf.fit(x_train, y_train)   
    print(kernel_, clf.score(x_train, y_train), clf.score(x_test, y_test))
"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data[train_columns], train_data['target'], stratify=train_data['target'])

import lightgbm as lgb

params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse', 'mae', 'mse'},
            'subsample': 0.2,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'num_leaves': 512,
            'max_depth': 10
            }

lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
lgb_eval = lgb.Dataset(x_test, y_test, categorical_feature=categoricals)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=100,
               verbose_eval = 200)
"""