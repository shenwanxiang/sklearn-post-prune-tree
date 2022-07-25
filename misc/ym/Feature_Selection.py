#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:19:34 2017

@author: charleshen


A essimble feature selection method, Stepwise Feature Elimination,

based on feature importance(rank) methods


Reference:
    Shen W, Xiao T, Chen S, et al. 
    Predicting the Enzymatic Hydrolysis Half-lives of New Chemicals 
    Using Support Vector Regression Models Based on Stepwise Feature Elimination[J]. 
    Molecular Informatics.

"""
#core module for parell and randomcv
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat,combinations
from collections import Iterable
from scipy.stats import mode,expon,randint 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#sklearns
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score,matthews_corrcoef, make_scorer

# core feature selection methods:
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, LassoLars, LassoLarsCV, RandomizedLasso
from sklearn import feature_selection
from sklearn.svm import SVC


from copy import deepcopy
from IPython.display import Image
from operator import itemgetter
from time import time

from DTGetBestParas import GBP
#import get_optimal_paras

import warnings
warnings.filterwarnings("ignore")

  
def inter_union_diff(list_of_list):
    inter = list_of_list[0]
    union = list_of_list[0]
    diff = list_of_list[0]
    for ls in list_of_list:
        inter = list(set(inter) & set(ls))
        union = list(set(union) | set(ls))
        diff =  list(set(diff) ^ set(ls))
    return inter, union, diff


def select_features(df, n_features = 40):
    select_list = []
    for col in df.columns:
        df1 = df[col].sort_values(ascending = False)
        select_list.append(df1.iloc[:n_features].index)
    inter, union, diff = inter_union_diff(select_list)
    return inter, union, diff


def fscore_im(df):
    
    '''
    F-score importance 
    
    input:
        df: dataframe, last column is label  
        
    Reference:
        
        Combining SVMs with Various Feature Selection Strategies,
        Yi-Wei Chen and Chih-Jen Lin
    '''
    
    dfy = df[df.columns[-1]]
    dfx = df[df.columns[:-1]]

    feature_names = dfx.columns
    
    X = dfx.values   
    y = dfy.values

    if len(np.unique(y)) >= 5:
        F,p = feature_selection.f_regression(X,y)
    else:
        F,p = feature_selection.f_classif(X,y)
    
    fscore = pd.DataFrame(F,index = feature_names,columns = ['fscore'])
    return fscore.sort_values('fscore',ascending = False)
    


def lasso_im(df,n_jobs = -1):
    '''
    Lasso Rigression methond based feature importance
    
    input:
        df: dataframe, last column is label
    
    '''
    dfy = df[df.columns[-1]]
    dfx = df[df.columns[:-1]]

    feature_names = dfx.columns
    
    X = dfx.values   
    y = dfy.values
    
    model_lasso = LassoCV(eps=0.0000001, 
                          n_alphas=200, 
                          max_iter=10000, 
                          cv=5, 
                          n_jobs = n_jobs,
                          precompute=True)
    model_llcv = LassoLarsCV(precompute='auto', 
                             max_iter=5000, 
                             verbose=False, 
                             cv=5, 
                             max_n_alphas=5000, 
                             n_jobs=n_jobs)
    model_lasso.fit(X, y)
    model_llcv.fit(X, y)    

    coef1 = pd.DataFrame(model_lasso.coef_, columns = ['lassocv'])  
    coef2 = pd.DataFrame(model_llcv.coef_, columns = ['lassolarcv'])
    
    coef1.index = feature_names
    coef2.index = feature_names
    
    dflasso = coef1.join(coef2).mean(axis = 1)
    dflasso = dflasso.to_frame(name='lasso')
    dflasso = dflasso.sort_values('lasso', ascending=False)    
    
    return dflasso




def elnet_im(df,n_jobs=-1):
    '''
    ElasticNetCV methond based feature importance
    
    input:
        df: dataframe, last column is label
    
    '''
    dfy = df[df.columns[-1]]
    dfx = df[df.columns[:-1]]

    feature_names = dfx.columns
    
    X = dfx.values   
    y = dfy.values
    
    model_elnet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99],
                               eps=0.0000001, 
                               n_alphas=100, 
                               max_iter=50000, 
                               cv=5, 
                               verbose=False, 
                               precompute=True, 
                               n_jobs=n_jobs)
    model_elnet.fit(X, y)
    coef1 = pd.DataFrame(model_elnet.coef_, columns = ['elnet'])  
    coef1.index = feature_names
    coef1 = coef1.sort_values('elnet',ascending = False)



def pcc_im(df):
    '''
    
    Person coef importance:
        
    input:
        df: dataframe, last column is label
    '''
    
    pcc = df.corr()[df.corr().columns[-1]]
    pcc = pcc[:-1]
    s = pcc.sort_values(ascending = False)
    return s.to_frame(name = 'pcc')
    


def tree_im(df, n_jobs= -1):
    
    '''
    Optimal tree method based feature importance
    
    input:
        df: dataframe, last column is label
    
    use the best paras to find feature_importance
    
    '''
    #get best paras
    
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
    feature_names = df.columns[:-1]
    
    op = GBP(method='both')
    
    op.fit(X,y)
        
    #use the paras get feature importance
    tree_ims = TreeImpFeat(X,y, 
                                max_depth=op.max_depth,
                                min_samples_leaf=op.min_samples_leaf,
                                min_samples_split=op.min_samples_split,
                                max_leaf_nodes=op.max_leaf_nodes,
                                feature_names=feature_names
                                )
    
    
    return tree_ims.to_frame(name = 'tree') 
        


def get_order_list(feat_seq):
    all_combins = [feat_seq[i:j] for i, j in combinations(range(len(feat_seq)+1), 2)]
    return all_combins[:len(feat_seq)]


def get_feat_list(orig_feat, pool_feat):
   return [orig_feat + [i] for i in pool_feat]
 


def multip_run(fuction,task_zip,n_jobs = 3):
    pool = ThreadPool(processes=n_jobs)
    results  = pool.starmap(fuction, task_zip)
    pool.close()
    pool.join()
    return results 


def randomgridsearch(X,y, estimator,param_distributions, 
                     n_jobs = -1, cv = 3, n_iter = 5, randcv = True):

    
    if len(np.unique(y)) <= 10:
        scoring = 'roc_auc'
    else: scoring = 'r2'
    
    if randcv:
        clf = RandomizedSearchCV(estimator, param_distributions,
                                 n_iter=n_iter,n_jobs=n_jobs,cv =cv,
                                 scoring = scoring)
    else:
        clf = GridSearchCV(estimator, param_distributions, 
                           n_jobs=n_jobs,
                           cv =3,
                           scoring = scoring)
                           
    clf.fit(X, y)        
    return clf.best_score_,clf.best_params_



def cvscore(X,y,estimator,best_paras):
    
    if len(np.unique(y)) <= 10:
        scoring = 'roc_auc'
    else: scoring = 'r2'
    
    clf = estimator.set_params(**best_paras)
    score = cross_val_score(clf, X,y,cv=3,n_jobs = -1,scoring = scoring)
    return score.mean()
    
    
def TreeImpFeat(X,y, seed_list=list(range(0,500,50)),**kwargs):
    '''
    input:
        X,y: array
        kwargs: dict of paras in tree, can be empty

    output: 
        feature_importance    
    '''
    #X = df[df.columns[:-1]].values
    #y = df[df.columns[-1]].values
    
    if 'feature_names' in kwargs:
        feature_names = kwargs['feature_names']
        if len(feature_names) == X.shape[1] and isinstance(feature_names, Iterable):
            pass
        else:
            print('feature_names is not iterable or not consitence with the shape of X')
            feature_names = range(X.shape[1]) 
    else:
        feature_names = range(X.shape[1])
    
    
    if len(np.unique(y)) > 5:
        clf = tree.DecisionTreeRegressor()
        
        if 'max_depth' in kwargs:
            clf.max_depth = kwargs['max_depth']
        if 'max_leaf_nodes' in kwargs:
            clf.max_leaf_nodes = kwargs['max_leaf_nodes']
        if 'min_samples_leaf' in kwargs:
            clf.min_samples_leaf = kwargs['min_samples_leaf']
    
        dfall =pd.DataFrame(index= feature_names)
          
        for i in seed_list:
            clf.random_state = i
            
            clf2 = tree.DecisionTreeRegressor(random_state=clf.random_state,
                                              max_depth=clf.max_depth,
                                              max_leaf_nodes=clf.max_leaf_nodes,
                                              min_samples_leaf=clf.min_samples_leaf)
            clf2.fit(X,y)
            coef = pd.DataFrame(clf2.feature_importances_,columns = [i],index= feature_names)
            dfall = dfall.join(coef)
            
    else:
        clf = tree.DecisionTreeClassifier()
        
        if 'max_depth' in kwargs:
            clf.max_depth = kwargs['max_depth']
        if 'max_leaf_nodes' in kwargs:
            clf.max_leaf_nodes = kwargs['max_leaf_nodes']
        if 'min_samples_leaf' in kwargs:
            clf.min_samples_leaf = kwargs['min_samples_leaf']
    
        dfall =pd.DataFrame(index= feature_names)

        for i in seed_list:
            clf.random_state = i
            
            clf2 = tree.DecisionTreeClassifier(random_state=clf.random_state,
                                              max_depth=clf.max_depth,
                                              max_leaf_nodes=clf.max_leaf_nodes,
                                              min_samples_leaf=clf.min_samples_leaf)
            clf2.fit(X,y)
            coef = pd.DataFrame(clf2.feature_importances_,columns = [i],index= feature_names)
            dfall = dfall.join(coef)                
    dfall.columns.name = 'random_seed'
    
    return dfall.mean(axis = 1).sort_values(ascending = False)


def SFE(df, estimator, param_grid, im_method = 'seq', sel_method = 'sfe',
        vip_feat = 20, Forward = True, max_feat = 50, batch_feat = 1,
        para_search_step = 1000, random_state = None, n_jobs = 4,
        cv = 5, randcv = True):
    
    dfy = df[df.columns[-1]]
    dfx = df[df.columns[:-1]]
    feature_names = dfx.columns
    
    if im_method == 'fscore':
        dfs = fscore_im(df)
        dfs = abs(dfs)
        dfs = dfs.sort_values('fscore',ascending = False)
    
    elif im_method == 'pcc':
        dfs = pcc_im(df)
        dfs = abs(dfs)
        dfs = dfs.sort_values('pcc',ascending = False)   
        
    elif im_method == 'tree':
        dfs = tree_im(df)
    
    elif im_method == 'lasso':
        dfs = lasso_im(df)
        dfs = abs(dfs)
        dfs = dfs.sort_values('lasso',ascending = False)  
        
    elif im_method == 'elnet':
        dfs = elnet_im(df)
        dfs = abs(dfs)
        dfs = dfs.sort_values('elnet',ascending = False)          
    elif im_method ==  'seq':
        dfs = pd.DataFrame(list(range(len(feature_names))), index=feature_names,columns = ['seq'])
    
    else: print("unknow method, methods of 'fscore','pcc','tree','lasso','elnet','seq' are supported")

    if sel_method == 'sfe':
        print('Finish calculating feature importance, SFE is performancing......')
        
        orig_feat = list(dfs[:vip_feat].index)
        
        pool_feat = list(dfs[vip_feat:].index)
    
        X0 = dfx[orig_feat].values
        y = dfy.values
        
        best_score,best_paras = randomgridsearch(X0,y, estimator,param_grid, n_jobs = -1, randcv = randcv, cv = cv)
        
        score_list = []
        score_list.append(best_score)
        
        myloop = int((max_feat-len(orig_feat))/batch_feat)
        
        change_flag = para_search_step
        
        for i in range(myloop):
            
            start = time()
    
            change_flag = change_flag-1
            
            if change_flag == 0:
                change_flag = para_search_step
                print('filter features and re-search best paras, please wait ...')
                best_score,best_paras = randomgridsearch(dfx[orig_feat].values,y, estimator,param_grid, n_jobs = -1,randcv = randcv, cv = cv)
                    
            try_feat = get_feat_list(orig_feat, pool_feat)   
            Xs = [df[i].values for i in try_feat]
            n = len(Xs)
            task_zip = zip(Xs,repeat(y,n),repeat(estimator,n),repeat(best_paras,n))
            #multi-porcessing run using n_jobs
            
            onescore = multip_run(cvscore,task_zip,n_jobs = n_jobs)
            dfonescore = pd.DataFrame(onescore,index = pool_feat,columns = ['score'])
            
            selected_feat = dfonescore.sort_values('score',ascending = False).iloc[:batch_feat].index
            selected_score = dfonescore.sort_values('score',ascending = False).iloc[:batch_feat].values
            
            for f in list(selected_feat):
                orig_feat.append(f)
                pool_feat.remove(f)
            for s in selected_score:
                score_list.append(s[0])
                
            end = time()
            interval = end-start
            print('feature_size:%d,' % len(orig_feat),'time used: %.3f second' % interval, 'cv_score: %f,' % best_score,'paras: %s' % best_paras)
                
        f_score = pd.DataFrame(score_list,index=range(vip_feat, len(score_list)+vip_feat),columns=['cv_score'])
        f_sequ = pd.DataFrame(orig_feat,index = range(1,len(orig_feat)+1),columns = ['feature_names'])
        return f_sequ.join(f_score)
    
    if sel_method == 'ofe':
        print('Finish calculating feature importance, OFE is performancing......')
        
        all_features = list(dfs.index)
        
        try_feat = get_order_list(all_features[:max_feat])
        
        Xs = [df[i].values for i in try_feat]
        
        y = dfy.values
        
        n= len(Xs)
        task_zip = zip(Xs,repeat(y,n),repeat(estimator,n),
                       repeat(param_grid,n),repeat(1,n),
                       repeat(3,n),repeat(5,n),repeat(True,n))
        
        myre = multip_run(randomgridsearch,task_zip,n_jobs = n_jobs)
        
        dfre = pd.DataFrame(myre,columns = ['cv_score','paras']).join(dfs.reset_index(drop = False))
        
        return dfre

if __name__ == "__main__":                                        
    #df = pd.read_csv('../data/bairong_train.csv',index_col = 'uid')
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    df = pd.DataFrame(X,columns=boston.feature_names).join(pd.DataFrame(y,columns=['target']))
    #estimator = SVC(cache_size=7000,gamma = 0.1)
    #param_grid = {'C': expon(scale=100)}
    #param_grid = {'C':[1,100,1000],'gamma':[0.1, 0.5]}
    #param_grid = {'C': expon(scale=100), 'gamma': expon(scale=.1)}  
    

    estimator = tree.DecisionTreeRegressor(max_leaf_nodes = 10,min_samples_leaf = 5)
    param_grid = {'max_depth': randint(5,20)}
   
    results = SFE(df, estimator, param_grid, im_method = 'tree', sel_method = 'ofe',
                  vip_feat = 1, Forward = True,max_feat = 142,batch_feat = 2,
                  para_search_step = 5, random_state = None,n_jobs = 4,cv =3)
    
    results['cv_score'].plot(figsize=(16,8), fontsize = 20,lw = 3)
    plt.xlabel('Number of feature selected', fontsize = 30)
    plt.ylabel('Cross-validated score', fontsize = 30)    
    
    
    
    
    
    
    
    