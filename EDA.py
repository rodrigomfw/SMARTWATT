# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
sns.set_style('darkgrid')

if __name__ == '__main__':
    
    id = '04'
    
    train = pd.read_csv('data/id_'+id+'_train.csv', sep=';')
    test = pd.read_csv('data/id_'+id+'_test.csv', sep=';')
    if id == '51':
        train = train[1454:]   
        
    # Feature Engineering
    train['hour'] = train['predictiondate'].apply(lambda x: int(x[11:13]))
    train['day'] = train['predictiondate'].apply(lambda x: int(x[:2]))
    train['month'] = train['predictiondate'].apply(lambda x: int(x[3:5]))
    test['hour'] = test['predictiondate'].apply(lambda x: int(x[11:13]))
    test['day'] = test['predictiondate'].apply(lambda x: int(x[:2]))
    test['month'] = test['predictiondate'].apply(lambda x: int(x[3:5]))

    # Missing data
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print missing_data.head(15)

    # Feature Engineering
    train['hour'] = train['predictiondate'].apply(lambda x: int(x[11:13]))
    train['day'] = train['predictiondate'].apply(lambda x: int(x[:2]))
    train['month'] = train['predictiondate'].apply(lambda x: int(x[3:5]))
 
    train.drop(['predictiondate', 'wind_gust', 'lightning_risk',
                'measured_wind_gust', 'availability'], axis=1, inplace=True)
#    train['measured_wind_direction'].fillna(0, inplace=True)

    print train['Power'].describe()
    
    # Data Visualization
    plt.figure()
    sns.boxplot(x='hour', y="Power", data=train)
    plt.ylabel('Power (kW)')
    plt.xlabel('Hour')

    plt.figure()
    sns.boxplot(x='day', y="Power", data=train)
    plt.ylabel('Power (kW)')
    plt.xlabel('Day')
    
    plt.figure()
    sns.boxplot(x='month', y="Power", data=train)
    plt.ylabel('Power (kW)')
    plt.xlabel('Month')
    
    plt.figure()
    train.plot.scatter(x='wind_speed', y='Power')
    plt.ylabel('Power (kW)', fontsize=13)
    plt.xlabel('Wind Speed (m/s)', fontsize=13)
    
    plt.figure()
    train.plot.scatter(x='wind_direction', y='Power')
    plt.ylabel('Power(kW)', fontsize=13)
    plt.xlabel('Wind Direction ($^\circ$)', fontsize=13)

    plt.figure()
    train.plot.scatter(x='measured_wind_speed', y='Power')
    plt.ylabel('Power(kW)', fontsize=13)
    plt.xlabel('Measured Wind Speed (m/s)', fontsize=13)

    plt.figure()
    train.plot.scatter(x='measured_wind_direction', y='Power')
    plt.ylabel('Power(kW)', fontsize=13)
    plt.xlabel('Measured Wind Direction ($^\circ$)', fontsize=13)

    plt.figure()
    train.plot.scatter(x='precipitation', y='Power')    
    plt.ylabel('Power(kW)', fontsize=13)
    plt.xlabel('Precipitation (mm)', fontsize=13)

    plt.figure()
    train.plot.scatter(x='temp', y='Power')
    plt.ylabel('Power(kW)', fontsize=13)
    plt.xlabel('Temperature ($^\circ$C)', fontsize=13)
    
    plt.figure()
    train.plot.scatter(x='wind_speed', y='measured_wind_speed')
    plt.ylabel('Measured Wind Speed (m/s)', fontsize=13)
    plt.xlabel('Wind Speed (m/s)', fontsize=13)

    plt.figure()
    train.plot.scatter(x='wind_direction', y='measured_wind_direction')
    plt.ylabel('Measured Wind Direction ($^\circ$)', fontsize=13)
    plt.xlabel('Wind Direction ($^\circ$)', fontsize=13)
    
    sns.heatmap(train.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
    plt.show()
    
    plt.figure()
    sns.distplot(train['Power'])
    plt.figure()
    sns.distplot(train['wind_speed'])
    plt.figure()
    sns.distplot(train['wind_direction'])
    plt.figure()
    sns.distplot(train['precipitation'])
    plt.figure()
    sns.distplot(train['temp'])
    
    # Feature selection
    features = ['wind_speed', 'wind_direction', 'temp',
                'precipitation', 'hour', 'month']
    max_power = train['Power'].max()
    train['Power'] = train['Power'].divide(max_power)   
    train, test = train_test_split(train, train_size=0.85, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].values)
    y_train = train['Power'].divide(train['Power'].max()).values
        
    model = lgb.LGBMRegressor()
    rfe = RFECV(estimator=model, cv=10, step=1)
    rfe = rfe.fit(X_train, y_train)
    print rfe.support_
    
    model = xgb.XGBRegressor()
    rfe = RFECV(estimator=model, cv=10, step=1)
    rfe = rfe.fit(X_train, y_train)
    print rfe.support_
