#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:24:20 2017

@author: charleshen
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error
import FillNaN

from sklearn.tree import tree

from sklearn.tree import tree_prune
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



import DTGetBestParas as GBP

'''
df = pd.read_csv('miss_data/1.csv',index_col = 'id')

for x in df.columns[df.dtypes == 'object']:
    df[x] = df[x].str.strip()
    df[x] = df[x].str.replace(',','-').str.replace(' ','').str.replace('\t','-').str.replace(';','-')
df.columns = df.columns.str.strip().str.replace(',','-').str.replace(' ','')
       
df_train,df_test = train_test_split(df, test_size=0.25, random_state=42) 
df_train.to_csv('miss_data/df_train.csv')
df_test.to_csv('miss_data/df_test.csv')






l = FillNaN.FMD( method = 'ht', 
                 train_path = 'miss_data/df_train.csv',
                 test_path = 'miss_data/df_test.csv',   
                 #test_path = None,
                 
                 index_col = 'id',
                 target_col = 'BinaryTarget',
                 n_jobs = 8,
                 random_state = 17,
                 dtree_method = 'both',
                 
                 distence_cater = 'rogerstanimoto',
                 distence_numer = 'correlation',

                 weight_cater = None,
                 weight_numer = None,

                 htree_method = 'complete',
                 htree_features = 'tree',
                 htree_dis_start = 0.1,
                 htree_dis_step = 0.1)


tr1,ts1 = l.fit()




l = FillNaN.FMD( method = 'dt', 
                 train_path = 'miss_data/df_train.csv',
                 test_path = 'miss_data/df_test.csv',   
                 #test_path = None,
                 
                 index_col = 'id',
                 target_col = 'BinaryTarget',
                 n_jobs = 8,
                 random_state = 17,
                 dtree_method = 'both',
                 
                 distence_cater = 'rogerstanimoto',
                 distence_numer = 'correlation',

                 weight_cater = None,
                 weight_numer = None,

                 htree_method = 'complete',
                 htree_features = 'tree',
                 htree_dis_start = 0.1,
                 htree_dis_step = 0.1)


tr2,ts2 = l.fit()


'''



train_list =['miss_data/fill/df_train.csv',
             'miss_data/fill/dt_df_train.csv',
             'miss_data/fill/ht_df_train.csv']



test_list =['miss_data/fill/df_test.csv',
             'miss_data/fill/dt_df_test.csv',
             'miss_data/fill/ht_df_test.csv']


score_list =[]

for tr,ts in zip(train_list,test_list):
    l = FillNaN.FMD(train_path=tr,test_path=ts)
    
    prep_df_train = l.load_train_data_prep()
    prep_df_test =  l.load_test_data_prep()

    
    dt = tree_prune.DecisionTreeClassifier(random_state=17).fit(l.X_train,l.y_train)
    pdt = GBP.DTGetBestParas(method = 'both').fit(l.X_train,l.y_train).clf
    gbdt = GradientBoostingClassifier().fit(l.X_train,l.y_train)
    rf = RandomForestClassifier().fit(l.X_train,l.y_train)
    

    auc_dt = roc_auc_score(l.y_test,dt.predict_proba(l.X_test)[:,1])
    auc_pdt = roc_auc_score(l.y_test,pdt.predict_proba(l.X_test)[:,1])    
    auc_gbdt = roc_auc_score(l.y_test,gbdt.predict_proba(l.X_test)[:,1])
    auc_rf = roc_auc_score(l.y_test,rf.predict_proba(l.X_test)[:,1])
    
    score_list.append([auc_dt,auc_pdt,auc_gbdt,auc_rf])

#[0.93888411690137918, 0.93940307054305983, 0.93888411690137918]

pd.DataFrame(score_list,
             columns =['auc_dt','auc_pdt','auc_gbdt','auc_rf'],
             index=train_list)

