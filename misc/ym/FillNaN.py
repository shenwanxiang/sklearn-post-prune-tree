#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

#python 3.6.0 linux

#python 3.6.3 win
Created on Mon Nov 13 10:40:53 2017

@author: charleshen

@usage: this is a module for filling missing values 
"""

print(__doc__)

import pandas as pd
import numpy as np

from DTGetBestParas import GBP
import tree_prune


from sklearn.metrics import roc_auc_score,mean_squared_error
from time import time

from sklearn.tree import tree


from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool as ThreadPool



#from multiprocessing import Pool as ThreadPool
from itertools import repeat
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import re

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from scipy.spatial.distance import pdist,squareform


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d,ls='--', c='r',lw=2)
    return ddata
        


import threading
import logging
import sys
import traceback

def map_parallel(f, iter, max_parallel = 10):
    """Just like map(f, iter) but each is done in a separate thread."""
    # Put all of the items in the queue, keep track of order.
    from queue import Queue, Empty
    total_items = 0
    queue = Queue()
    for i, arg in enumerate(iter):
        queue.put((i, arg))
        total_items += 1
    # No point in creating more thread objects than necessary.
    if max_parallel > total_items:
        max_parallel = total_items

    # The worker thread.
    res = {}
    errors = {}
    class Worker(threading.Thread):
        def run(self):
            while not errors:
                try:
                    num, arg = queue.get(block = False)
                    try:
                        res[num] = f(arg)
                    except Exception as e:
                        errors[num] = sys.exc_info()
                except Empty:
                    break

    # Create the threads.
    threads = [Worker() for _ in range(max_parallel)]
    # Start the threads.
    [t.start() for t in threads]
    # Wait for the threads to finish.
    [t.join() for t in threads]

    if errors:
        if len(errors) > 1:
            logging.warning("map_parallel multiple errors: %d:\n%s"%(
                len(errors), errors))
        # Just raise the first one.
        item_i = min(errors.keys())
        type, value, tb = errors[item_i]
        # Print the original traceback
        logging.info("map_parallel exception on item %s/%s:\n%s"%(
            item_i, total_items, "\n".join(traceback.format_tb(tb))))
        raise value
    return [res[i] for i in range(len(res))]


#l = FMD()

class FMV(object):
    '''
    filling missing data using tree groups
    including: Decision tree groups and Hierarchical tree groups
    
    method of dt for: Decision Tree methods
    method of ft for: Hierarchical Tree methods
        
    '''
    
    def __init__(self, 
                 method = 'dt', 
                 train_path = 'dataset/df_train.csv',
                 test_path = 'dataset/df_test.csv',   
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
                 htree_dis_step = 0.1,
                 
                 **kwargs):
        
        #basic info.        
        self.method = method
        self.n_jobs =n_jobs
        self.data_path = train_path
        self.test_path = test_path
        self.index_col = index_col
        self.target_col = target_col
        
        #parameters for dt
        self.random_state = random_state
        self.dtree_method = dtree_method


        #distence and weight for ht      
        self.distence_cater =  distence_cater 
        self.distence_numer = distence_numer
    
        
        self.weight_cater = weight_cater
        self.weight_numer = weight_numer
        self.htree_features = htree_features        
        self.htree_method = htree_method
        self.htree_dis_start = 0.1
        self.htree_dis_step = 0.1

        
        try:
            for key, value in kwargs.items():
                if key != None:
                    setattr(self, key, value)
                else: pass
        except: pass        
        
        
    
    def fuc_get_dumps(self,df):
        
        df2 = pd.get_dummies(df,columns=df.columns)
        df2 = df2.astype('int64')
        
        return df2
    
    
    
    def fuc_map_string_to_num(self,df,unique = 6):
        
        dfreturn = pd.DataFrame(index = df.index,columns = df.columns)
        for c in  df.columns:
            one = df[c].unique()
            if one.shape[0] <= unique:
                onedict = pd.DataFrame(list(range(one.shape[0])),index = one).to_dict()[0]
                dfreturn[c] = df[c].map(onedict)
            else:
                dfreturn[c] = df[c]
        return dfreturn

            
    def fuc_remove_low_variance_features(self, data_frame,var = 0.8):
        
        '''
        get the 0.8 variance features by fitting VarianceThreshold
        remove some feature who's variance that is below 0.8
        '''
        selector = VarianceThreshold(var)
        selector.fit(data_frame)
        # Get the indices of zero variance feats
        feat_ix_keep = selector.get_support(indices=True)
        orig_feat_ix = np.arange(data_frame.columns.size)
        feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
        # Delete zero variance feats from the original pandas data frame
        data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                     axis=1)
        return data_frame        
        

    
    def fuc_get_catergory_col(self, df, unique = 2):
        catergory = []
        for x in df.columns:
            if df[x].unique().shape[0] <= unique:
                catergory.append(x)
        return catergory
    
     
    def fuc_get_continuous_col(self, df, catergory_list):
        return [x for x in df.columns if not x in catergory_list] 
    
    
    def fuc_get_data_type_dict(self, catergory_list, continuous_list):
        data_type_dict = {}
        for n in catergory_list:
            data_type_dict[n] = 'catergory'
        for o in continuous_list:
            data_type_dict[o] = 'continuous'
        return data_type_dict
                
        
    def fuc_fillnan(self,dfx,dfy):
        
        catergory_list = self.fuc_get_catergory_col(dfx,unique=10)
        continuous_list = self.fuc_get_continuous_col(dfx,catergory_list)         
        
        imp1 = Imputer(missing_values=np.nan , strategy='most_frequent', axis=0)
        
        imp2 = Imputer(missing_values=np.nan , strategy='mean', axis=0)
        
        imp1 = imp1.fit(dfx[catergory_list],dfy)
        imp2 = imp2.fit(dfx[continuous_list],dfy)        
        
        dfn = pd.DataFrame(imp1.transform(dfx[catergory_list]),
                           index = dfx.index,
                           columns = catergory_list)
        
        dfo = pd.DataFrame(imp2.transform(dfx[continuous_list]),
                           index = dfx.index,
                           columns = continuous_list)   
        
        return dfn.join(dfo), imp1, imp2, catergory_list,continuous_list

        

    def fuc_fillnan_transform(self,dfx):
        
        imp1 = self.imputer1
        imp2 = self.imputer2
        
        catergory_list = self.catergory_list
        continuous_list  = self.continuous_list
         
        dfn = pd.DataFrame(imp1.transform(dfx[catergory_list]),
                           index = dfx.index,
                           columns = catergory_list)
         
         
        dfo = pd.DataFrame(imp2.transform(dfx[continuous_list]),
                           index = dfx.index,
                           columns = continuous_list)   
        return dfn.join(dfo)

        
    
    def load_train_data_prep(self):  
        
        '''
        load and prepare data
        
        '''
        
        df = pd.read_csv(self.data_path,index_col = self.index_col)
        self.original_df_train = df        


        for x in df.columns[df.dtypes == 'object']:
            df[x] = df[x].str.strip()
            df[x] = df[x].str.replace(',','-').str.replace(' ','')
        df.columns = df.columns.str.strip().str.replace(',','-').str.replace(' ','')
 

        
        if 'object' in df.dtypes.values:
            
            #dumpies object and remove many
            print('Object training data, please convert to numeric data First!')
            
            print('Otherwise, we will do one-hot decode automaticlly')
            
            df1 = df[df.columns[df.dtypes == 'object']]
            
            u = df1.describe().loc['unique']
            cols = u[u<=30]
            
            df11 = self.fuc_get_dumps(df1[cols.index])
            
            df2 =  df[df.columns[df.dtypes != 'object']]
            dffinal = df11.join(df2)
            
        else:
            dffinal = df
        
        
        self.feature_col = dffinal.columns[dffinal.columns != self.target_col]
        dfx = dffinal[self.feature_col]   
        dfy = dffinal[[self.target_col]]
        
        
        dfxx,imp1, imp2, catergory_list,continuous_list  = self.fuc_fillnan(dfx,dfy)
        
        
        self.imputer1 = imp1
        self.imputer2 = imp2
        
        self.catergory_list = catergory_list
        self.continuous_list = continuous_list
        
        n = len(dfy)
        
        self.X_train = dfxx.values
        self.y_train = dfy.values.reshape(n,)
        
        df_train = dfxx.join(dfy)
        
        return df_train 
        
    
    def load_test_data_prep(self):
        
        if self.test_path == None:
            df_test = pd.DataFrame([])
            self.X_test = None
            self.y_test = None
            self.original_df_test = df_test
            
        else:
            df = pd.read_csv(self.test_path, index_col = self.index_col)
            self.original_df_test = df                
            
            for x in df.columns[df.dtypes == 'object']:
                df[x] = df[x].str.strip()
                df[x] = df[x].str.replace(',','-').str.replace(' ','')
            df.columns = df.columns.str.strip().str.replace(',','-').str.replace(' ','')
            
            if 'object' in df.dtypes.values:
                
                #dumpies object and remove many
                print('Object test data, please convert to numeric data First!')
                
                print('Otherwise, we will do one-hot decode automaticlly')
                
                df1 = df[df.columns[df.dtypes == 'object']]
                
                u = df1.describe().loc['unique']
                cols = u[u<=30]
                
                df11 = self.fuc_get_dumps(df1[cols.index])
                
                df2 =  df[df.columns[df.dtypes != 'object']]
                dffinal = df11.join(df2)
                
            else:
                dffinal = df
            
            test_feature_col = dffinal.columns[dffinal.columns != self.target_col]
            dfx = dffinal[test_feature_col]   
            
            try:
                dfy = dffinal[[self.target_col]]
            except: 
                dfy = pd.DataFrame([],index = dfx.index,columns=[self.target_col])
            
            
            def add_missing_dummy_columns(d, columns):
                missing_cols = set( columns ) - set( d.columns )
                for c in missing_cols:
                    d[c] = 0            
            
            
            def fix_columns(d, columns):  
            
                add_missing_dummy_columns(d, columns)
            
                # make sure we have all the columns we need
                assert( set( columns ) - set( d.columns ) == set())
            
                extra_cols = set( d.columns ) - set( columns )
                if extra_cols:
                    print("extra columns:", extra_cols)
                d = d[ columns ]
                return d

            #fix test columns to train columns,namely feature_col
            fixed_dfx = fix_columns(dfx.copy(), self.feature_col)
            
            #then fill nan transform as train 
            dfxx = self.fuc_fillnan_transform(fixed_dfx)
            
            n = len(dfy)
            self.X_test = dfxx.values
            self.y_test = dfy.values.reshape(n,)
            
            df_test = dfxx.join(dfy)
            
        return df_test
        
        
        
    def main_dt_rules(self):  

        self.prep_df_train = self.load_train_data_prep()
        self.prep_df_test =  self.load_test_data_prep()
        
        ss = GBP(method = self.dtree_method,
                 n_jobs= self.n_jobs,
                 figtitle = self.data_path)
                                
        ss.fit(self.X_train,self.y_train)
        
        clf = ss.clf
        
        dot_data0 = tree_prune.export_graphviz(clf,
                                               out_file=None,
                                               feature_names=self.feature_col)
        
        self.clf = clf
        
        ls = dot_data0.split('\n')
        
        rules = []
        for s in ls:
            if re.findall(r"label=\"(.+?)\\",s):
                rules.append(re.findall(r"label=\"(.+?)\\",s)[0])    
        
        return rules
    


    def main_dt_dfrules(self, rules:list, df_train:pd.DataFrame):
        
        if df_train.empty:
            dfreules = pd.DataFrame()

        else:
            check_rule = [r.split(' <= ') for r in rules]
            
            dfreules = pd.DataFrame(columns = rules)
    
            for i,j in zip(check_rule, rules):
                dfreules[j] = df_train[i[0]] <= float(i[1])
            
        return dfreules
    
    
    def main_dt_group_groupby(self, dfrules, less=0):
                    
        grouplist = list(dfrules.columns)
        
        if less == 0:
            grouplist = grouplist
            
        else:
            grouplist=grouplist[:less]
            if less > 0:
                print('Warning: less should be less than 0')   
            else:pass
        
        groups = dfrules.groupby(grouplist)
        
        group_dict = groups.groups    
        
        return group_dict
    
                      
    def main_ht_get_Z(self, dfx_prep):
        
        cater_list = self.fuc_get_catergory_col(dfx_prep,unique=3)
        
        bool_in = dfx_prep.columns.isin(cater_list)
        
        numer_list = dfx_prep.columns[~bool_in]
        
        
        cater_p = len(cater_list)/len(dfx_prep.columns)
        numer_p = len(numer_list)/len(dfx_prep.columns)


        #@update the weight for default value
        if self.weight_cater is None:
            self.weight_cater = round(cater_p,3)
        else: pass
    
        if self.weight_numer is None:
            self.weight_numer = round(numer_p,3)
        else: pass
        
        
        df_cater = dfx_prep[cater_list]
        df_numer = dfx_prep[numer_list] #
        
        D_cater = pdist(df_cater,self.distence_cater)
        D_numer = pdist(df_numer,self.distence_numer)
        
        D_cater = pd.DataFrame(D_cater).fillna(0).values.reshape(D_cater.shape)
        D_numer = pd.DataFrame(D_numer).fillna(0).values.reshape(D_numer.shape)
        
        
        #scaled distence to 0-1
        if D_cater.max()-D_cater.min() == 0:
            D_cater_scaled = D_cater
            D_numer_scaled = (D_numer-D_numer.min())/(D_numer.max()-D_numer.min())
            
        elif D_numer.max()-D_numer.min() == 0:
            D_numer_scaled = D_numer
            D_cater_scaled = (D_cater-D_cater.min())/(D_cater.max()-D_cater.min())
            
        else:
            D_cater_scaled = (D_cater-D_cater.min())/(D_cater.max()-D_cater.min())
            D_numer_scaled = (D_numer-D_numer.min())/(D_numer.max()-D_numer.min())
        
        #jiaquan distence D: D = d1*w1 + d2*w2
        D = D_cater_scaled*self.weight_cater + D_numer_scaled* self.weight_numer

        Z= linkage(D,method=self.htree_method)
        
        # convert dedrogram
        #fig = plt.figure(figsize=(20,13)
        s = Z[:,2]
        k = (s-s.min())/(s.max()-s.min())
        Z[:,2] = k
        
        return D,Z
        
    def main_ht_plot_cluster(self, Z, class_num,figname='ht.png'):
        
        dis_thres = Z[:,2][-class_num-1]
       
        fig = plt.figure(figsize = (20,8))
        ax = fig.add_subplot(1, 1, 1)
        den = fancy_dendrogram(Z,
                         truncate_mode='lastp',
                         #labels = df.columns,
                         p = class_num,
                         color_threshold=dis_thres,                         
                         show_contracted=True,
                         leaf_rotation=90.,
                         leaf_font_size=3,                         
                         max_d= dis_thres, # plot a horizontal cut-off line
                         ax=ax)
        
        plt.title('Hierarchical Clustering Dendrogram (truncated)',fontsize=22)
        plt.xlabel('sample index or (cluster size)',fontsize=22)
        plt.ylabel('distance',fontsize=22)
        
        ax.tick_params(axis='x', which='major', labelsize=22)
        ax.tick_params(axis='y', which='major', labelsize=22)
        
        fig.savefig(figname)


    def main_ht_get_cluster(self, Z, max_d = 2):    

        clusters = fcluster(Z, max_d, criterion='distance')
        
        return clusters





    def OneGroupFill(self, dfall:pd.DataFrame, OneGroupDfIndex:pd.Index):
        '''
        Filling  miss values in one seperated group:dfonegroup,
        
        which came from dfall sliced by OneGroupDfIndex
        
        for object columns, use most_freq value,
        for int and float columns, use mean value
        
        '''
        dfonegroup = dfall.loc[OneGroupDfIndex]
        
        if dfonegroup.isnull().sum().sum():
            df_obj = dfonegroup.blocks.get('object')
            obj_dict = df_obj.describe().T.top.to_dict()
            df1 = df_obj.fillna(obj_dict)        
    
            bool_in = dfonegroup.columns.isin(df_obj.columns)
            
            df_numer = dfonegroup[dfonegroup.columns[~bool_in]] # ~ means logical negative
            numer_dict = df_numer.describe().T['mean'].to_dict()
            
            df2 = df_numer.fillna(numer_dict)
    
            df3 = df1.join(df2)
            
        else:
            df3 = dfonegroup
            
        return df3

    
    def MultiGroupRun(self, fuction, task_zip,n_jobs=6):

        #Multi-process Run

        pool = ThreadPool(processes=n_jobs)
        #pool.apply_async(fuction, task_zip)
        results  = pool.starmap(fuction, task_zip)

        pool.close()
        pool.join()
        
            
        return results 
    
    
    def fit(self):
        
        #method for descion tree
        
        
        df_train_new = self.load_train_data_prep()
        df_test_new = self.load_test_data_prep()
        
        df_train_old = self.original_df_train
        df_test_old = self.original_df_test
        
        if self.method == 'dt':
            
            #get rules list 
            rules = self.main_dt_rules()
            
            #diffrent in train and test because of index
            dfrules = self.main_dt_dfrules(rules, df_train_new) 
            
            self.rules = rules
            self.dfrules = dfrules
            

            
            #fill the missing values until No NaN
            dff = df_train_old.copy()
            i = 0        
            
            print(i, dff.isnull().sum().sum())
            while dff.isnull().sum().sum():

                
                
                train_group_dict = self.main_dt_group_groupby(dfrules,i)
                
                train_task_zip = zip(repeat(dff),train_group_dict.values())
                
                train_fill_lsdf = self.MultiGroupRun(self.OneGroupFill,train_task_zip,n_jobs=self.n_jobs)

                dff = pd.concat(train_fill_lsdf)               

                i=i-1   
                print(i, dff.isnull().sum().sum())
                
                if i == -len(rules):
                    break
            
            #convert to nummal index
            df_train_filled = pd.DataFrame([],index=df_train_old.index).join(dff)
            
            

            if df_test_new.empty:
                df_test_filled = pd.DataFrame()
            
            else:
                test_dfrules = self.main_dt_dfrules(rules, df_test_new)
                dff_t = df_test_old.copy()
                
                j = 0    
                print(j, dff_t.isnull().sum().sum())
                
                while dff_t.isnull().sum().sum():

                    test_group_dict = self.main_dt_group_groupby(test_dfrules,j)
                    
                    test_task_zip = zip(repeat(dff_t),test_group_dict.values())
                    
                    test_fill_lsdf = self.MultiGroupRun(self.OneGroupFill,test_task_zip,n_jobs=self.n_jobs)
    
                    dff_t = pd.concat(test_fill_lsdf)
                    
                   
                    
                    j=j-1    
                    print(j, dff_t.isnull().sum().sum()) 
                    
                    if j == -len(rules):
                        break
                
                df_test_filled = pd.DataFrame([],index=df_test_old.index).join(dff_t)
                #df_test_filled = dff_t
        
        
        #method for hireachical tree    
        elif self.method == 'ht':
            df_tofill = df_train_old.append(df_test_old)
            
            if self.original_df_test.empty:
                df_prep = df_train_new
            
            
            else:
            #wheather has a target column in test or not
                if self.target_col in  df_test_old.columns:
                    df_prep = df_train_new.append(df_test_new)
                    
                    
                    
                else:
                    new_col = list(df_train_new.columns)
                    new_col.remove(self.target_col)
                    df_prep = df_train_new[new_col].append(df_test_new[new_col])
              
            #for feature methods:
            if self.htree_features == 'tree':
                
                clf = tree.DecisionTreeClassifier().fit(self.X_train,self.y_train)
                self.feature_col_imp = clf.feature_importances_
                
                df_fimp = pd.DataFrame(clf.feature_importances_,
                                       index = self.feature_col,
                                       columns=['tree_imp']).sort_values('tree_imp',ascending = False)
                
                sel_cols = df_fimp[df_fimp['tree_imp'] > 0].index
                
                df_prep = df_prep[sel_cols]
                
                
            elif self.htree_features == 'prune_tree':
                rules = self.main_dt_rules()
                self.rules = rules
                sel_cols = [x.split(' <= ')[0] for x in rules]
                
                df_prep = df_prep[list(set(sel_cols))]
                
            else:
                df_prep = df_prep
                
            self.df_prep = df_prep
            D, Z  = self.main_ht_get_Z(df_prep)
            
            self.D = D            
            self.Z = Z
            
            
            dmax = Z[:,2].max()
            #dmin = Z[:,2].min()
            
            dis_start = self.htree_dis_start 
            
            dis_step = self.htree_dis_step
            
            g=0
            
            #c = df_tofill.shape[0]
            
            print(g, df_tofill.isnull().sum().sum(),round(dis_step,3), Z.shape[0]) 
            
            #for class_num in [int(c/4), int(c/8), int(c/10), 200, 1]:
            
            #plot
            self.main_ht_plot_cluster(Z, 80, figname='ht_80.png')               
            self.main_ht_plot_cluster(Z, 50, figname='ht_50.png')
            self.main_ht_plot_cluster(Z, 30, figname='ht_30.png')               
            self.main_ht_plot_cluster(Z, 10, figname='ht_10.png')            
            
            while df_tofill.isnull().sum().sum():
                
                clusters = self.main_ht_get_cluster(Z, dis_start)
                
                print('Number of classes are:', np.unique(clusters).shape[0])
                
                group_dict = pd.DataFrame(clusters, index= df_tofill.index).groupby(0).groups
                task_zip = zip(repeat(df_tofill),group_dict.values())
                
                fill_lsdf = self.MultiGroupRun(self.OneGroupFill,task_zip,n_jobs=self.n_jobs)
               
                #fill_lsdf = map_parallel(self.OneGroupFill, task_zip, max_parallel = self.n_jobs)
                
                df_tofill = pd.concat(fill_lsdf)
                
                
                
                dis_start = dis_start + dis_step
                
                g = g-1
                
                print(g, df_tofill.isnull().sum().sum(),round(dis_start,3),np.unique(clusters).shape[0]) 
                
                if dis_start > dmax:
                    break
            
            df_train_filled= pd.DataFrame([],index=df_train_old.index).join(df_tofill)
            df_test_filled= pd.DataFrame([],index=df_test_old.index).join(df_tofill) 
                
        else:
            print('method does not support, \'dt\',\'ht\' can be used' )
            df_train_filled = pd.DataFrame()
            df_test_filled = pd.DataFrame()
            
        return df_train_filled, df_test_filled




#for k,v in d0.items():
    
