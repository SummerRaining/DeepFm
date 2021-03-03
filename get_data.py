# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:57:50 2021

@author: shujie.wang
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime,timedelta
import numpy as np
import math

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline

def get_last_call_date(x,target_date = '2021-01-18'):
    #得到倒数第二次的拨打日期。
    if math.isnan(x):
        return float('nan')
    else:
        return (datetime.strptime(target_date,'%Y-%m-%d')+timedelta(days=-x)).strftime("%Y-%m-%d")
    
def get_last_call_days(x):
    if str(x['last_call_time'])=='nan' or str(x['stu_call_time']) == 'nan':
        return float('nan')
    else:
        return (datetime.strptime(x['stu_call_time'],'%Y-%m-%d')-datetime.strptime(x['last_call_time'],'%Y-%m-%d')).days

def feature_engineering1(df,target_date):
    #数据预处理
    df['stu_call_time'] = df.apply(lambda x:datetime.strftime(datetime.fromtimestamp(x['call_time_stamp']),'%Y-%m-%d'),axis = 1) #最后一次拨打时间。
    df['last_call_time'] = df['recent_cal_days'].apply(lambda x:get_last_call_date(x,target_date)) #倒数第二次的拨打时间。
    df['last_call_days'] = df.apply(get_last_call_days,axis = 1) #倒数第一次与倒数第二次拨打的间隔时间。
    df['bu_days'] = df['update_time'].map(lambda x:(datetime.strptime(target_date,'%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d')).days) #转入部门的天数。
    gen_feats = ['last_call_days','bu_days']
    return df,gen_feats
    
def get_data():
    sparse_cols = ['big_channel_name', 'small_channel_name', 'stu_city', 'stu_subject', 'student_channel',
                  'channel_code', 'grade_id', 'level','is_referral', 'verified', 'is_present_test_les', 
                  'high_transform_rate_city', 'high_connect_rate_city', 'good_grade', 'high_connect_rate_channel','high_transform_rate_channel']
    dense_cols = ['les_arrange_lesson_cnt', 'cal_cnt', 'connect_cnt', 'recent_cal_days', 'stu_multiple_registration_all',
                'cal_cnt_90', 'connect_cnt_90', 'stu_multiple_registration_all_90']
    
    data_path = '../data/'
    df = pd.read_csv(data_path+'call.txt','\t',error_bad_lines = False)
    df = df[~df['is_connect'].isna()]
       
    target_date = '2021-01-18'
    test_date = '2021-01-15'
    valid_date = '2021-01-12'
    #数据预处理   
    df,gen_feats = feature_engineering1(df,target_date)
    dense_cols += gen_feats
    #离散特征先补充缺失值，然后orderencoder
    df = df[sparse_cols+dense_cols+['is_connect','stu_call_time']]
    df[sparse_cols] = df[sparse_cols].fillna('missing_value').astype(str) 
    df[dense_cols] = df[dense_cols].fillna(df[dense_cols].median())
    
    
    numeric_transformer = Pipeline(
        steps=[
            ('standrd', StandardScaler())
        ]
    )
    categorical_transformer = Pipeline(
            steps=[
                ('category_encoder', OrdinalEncoder())
            ]
            )
    
    preprocessor = ColumnTransformer(
            transformers=[
                ('cat',categorical_transformer,sparse_cols),
                ('num',numeric_transformer,dense_cols)
                ]
        )#特征转换，离散特征顺序编码，连续特征标准化编码。
    
    _d = preprocessor.fit_transform(df) #拟合    
    df[sparse_cols+dense_cols] = _d
    
    sparse_max_len = list(df[sparse_cols].max().values+1)
    target_date = '2021-01-18'
    test_date = '2021-01-15'
    valid_date = '2021-01-12'
    #分割训练集和测试集
    train = df[(df.stu_call_time < valid_date)]
    valid = df[(df.stu_call_time >= valid_date) & (df.stu_call_time < test_date)]
    test = df[(df.stu_call_time >= test_date)]
    
    y = train['is_connect']
    x = train.drop(['is_connect','stu_call_time'], axis = 1)
    
    y_val = valid['is_connect']
    x_val = valid.drop(['is_connect','stu_call_time'], axis = 1)
    
    y_test = test['is_connect']
    x_test = test.drop(['is_connect','stu_call_time'], axis = 1)
    print(x.shape,x_val.shape,x_test.shape)
    return x,y,x_val,y_val,x_test,y_test,sparse_max_len

if __name__ =='__main__':
    x,y,x_val,y_val,x_test,y_test = get_data()