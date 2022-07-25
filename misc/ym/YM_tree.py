#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:33:03 2017

@author: charleshen
"""
#import three major modules

import pandas as pd
import numpy as np
from Feature_Selection import TreeImpFeat, SFE
from DTGetBestParas import GBP
from FillNaN2 import FMV

class YMTree(object):
    
    
    def __init__(self, df, target_col,
                 n_jobs=8,
                 method = 'both',
                 dtree_gbp = True,
                 strategy='dt'
                 ):
        
        self.fmv = FMV(n_jobs=n_jobs,
                       strategy=strategy,
                       dtree_method = method,
                       dtree_gbp = dtree_gbp).fit(df, target_col)
        
        self.records = self.fmv.records
        
        if strategy == 'dt':
            self.paras = self.fmv.paras 
        else:
            self.paras = dict()

    def impute(self, df):
        return self.fmv.transform(df)

    def get_paras(self, X,y,method='both'):
        '''
        inputs:
            X:numpy array,
            y:numpy array,
            method: str,'both','prune','tune','cal','none'
        
        return:
            dict of optimal parameters
        '''
        op = GBP(method=method)
        op.fit(X,y)  
        return dict(max_depth=op.max_depth,
                    min_samples_leaf=op.min_samples_leaf,
                    min_samples_split=op.min_samples_split,
                    max_leaf_nodes=op.max_leaf_nodes)
    
    
    def get_features(self, X,y, feature_names=None, alpha=0.05):
        '''
        inputs:
            X:numpy array,
            y:numpy array,
            feature_names: list of feature names
            alpha:threshold of selection VIP features(_feature_importance threshold)
        
        return:
            list of satisfied features,if feature_names is None, return the index
            
        '''
        #op = GBP(method='both')
        #op.fit(X,y)
        
        #use the paras get feature importance
        if feature_names is None:
            tree_ims = TreeImpFeat(X,y,
                                    #max_depth=op.max_depth,
                                    #min_samples_leaf=op.min_samples_leaf,
                                    #min_samples_split=op.min_samples_split,
                                    #max_leaf_nodes=op.max_leaf_nodes,
                                    )        
        else:
            tree_ims = TreeImpFeat(X,y, feature_names=feature_names
                                    #max_depth=op.max_depth,
                                    #min_samples_leaf=op.min_samples_leaf,
                                    #min_samples_split=op.min_samples_split,
                                    #max_leaf_nodes=op.max_leaf_nodes,
                                    )
        
        
        return tree_ims[tree_ims > alpha].index


if __name__ == "__main__":                                        
    #df = pd.read_csv('../data/bairong_train.csv',index_col = 'uid')
    
    from YM_tree import YMTree
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    df = pd.DataFrame(X,columns=boston.feature_names).join(pd.DataFrame(y,columns=['target']))

    T = YM_tree(df,'target')
    
    T.records
    T.paras
    
    df_filled = T.impute(df)
    
    
    T.get_paras(X,y)
    T.get_features(X,y,feature_names=boston.feature_names,alpha=0.01)
    