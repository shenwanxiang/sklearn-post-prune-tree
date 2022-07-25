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
from Feature_Selection import TreeImpFeat,SFE
import tree_prune
from sklearn.metrics import pairwise_distances

from sklearn.metrics import roc_auc_score,mean_squared_error
from time import time

from sklearn.tree import tree,export_graphviz

import pydotplus
from IPython.display import Image
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
    
    """Imputation transformer for completing missing values.
    
    filling missing data using tree groups
    including: Decision tree groups and Hierarchical tree groups


    Parameters
    ----------

    strategy : string, optional (default="dt")
        The imputation strategy.

        - If "dt", then replace missing values using the Decision Tree method.
        - If "ht", then replace missing values using the Hierarchical Tree method

    n_jobs : int, optional (default=8)
        The number of CPUs to use.

    random_state : int, optional (default=17)
        The random state for Decision Tree strategy.


    dtree_method : string, optional (default="both")
        The parameters for Decision Tree strategy.

        - If "both", then build the Decision Tree using best parameters.
        - If "prune", then build the Decision Tree using prune parameters.
        - If "tune", then build the Decision Tree using tune parameters.
        - If "cal", then build the Decision Tree using calculated parameters.
        - If "none", then build the Decision Tree using sklearn default parameters.

    distence_cater: string, optional (default="rogerstanimoto")
        The distence for catergory data

    distence_numer: string, optional (default="correlation")
        The distence for numeric data

    weight_cater: float or None, optional (default=None)
        The weight for catergory data

    weight_numer: float or None, optional (default=None)
        The weight for numeric data

    htree_method: str, optional (default='complete')
        The method for Hierarchical tree

    htree_features: str, optional (default='tree')
        The features for  Hierarchical tree to use
        
        - If "tree", then build the Hierarchical Tree using tree feature_importance which above 0.05.
        - If "prune_tree", then build the Hierarchical Tree using features of prune tree.
        
    htree_dis_start: float or None, optional (default=0.1)
        The distence for Hierarchical tree strategy to make started groups
        
    htree_dis_step: float or None, optional (default=0.1)
        The step distence for Hierarchical tree strategy to make coming groups, must above 0
    
                 
    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature if axis == 0.

    Notes
    -----
    - When ``htree_dis_start=0``, each sample is one group, the larger htree_dis_start is, the smaller number of groups is

    """
    
    def __init__(self, 
                 strategy = 'dt', 
                 #train_path = 'miss_data/df_train.csv',
                 #test_path = 'miss_data/df_test.csv',   
                 #test_path = None,
                 
                 #index_col = 'id',
                 #target_col = 'BinaryTarget',
                 n_jobs = 8,
                 random_state = 17,
                 dtree_method = 'both',
                 dtree_gbp = True,
                 
                 
                 distence_cater = 'rogerstanimoto',
                 distence_numer = 'correlation',
                 weight_cater = None,
                 weight_numer = None,

                 htree_method = 'complete',
                 htree_features = 'tree',
                 htree_features_alpha=0.01,
                 htree_dis_start = 0.2,
                 htree_dis_step = 0.15,
                
                 figtitle = 'FillNaN_figure',
                 **kwargs):
        
        #basic info.        
        self.strategy = strategy
        self.n_jobs =n_jobs
        self.figtitle = figtitle
        
        #parameters for dt
        self.random_state = random_state
        self.dtree_method = dtree_method
        self.dtree_gbp = dtree_gbp

        #distence and weight for ht      
        self.distence_cater =  distence_cater 
        self.distence_numer = distence_numer
    
        
        self.weight_cater = weight_cater
        self.weight_numer = weight_numer
        self.htree_features = htree_features        
        self.htree_method = htree_method
        
        self.htree_features_alpha = htree_features_alpha
        self.htree_dis_start = htree_dis_start
        self.htree_dis_step = htree_dis_step

        
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

        
    
    def fuc_prep_train_data(self, df, target_col):  
        
        '''
        load and prepare data
        
        '''
        
        #df = pd.read_csv(self.data_path,index_col = self.index_col)
        self.original_df_train = df        
        self.target_col = target_col
        self.variable_col = df.columns[df.columns != target_col]
        

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
        
    
    def fuc_prep_test_data(self, df):
        
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
        
        
        
    def main_dt_rules(self, X_train,y_train):  

        #self.prep_df_train = self.load_train_data_prep()
        #self.prep_df_test =  self.load_test_data_prep()
        
        ss = GBP(method = self.dtree_method,
                 n_jobs= self.n_jobs,
                 figtitle = self.figtitle)
                                
        ss.fit(X_train,y_train)

        paras = dict(max_depth=ss.max_depth,
                min_samples_leaf=ss.min_samples_leaf,
                min_samples_split=ss.min_samples_split,
                max_leaf_nodes=ss.max_leaf_nodes)
        
        clf = ss.clf
                
        if len(np.unique(y_train)) <= 5:
            clf_skl = tree.DecisionTreeClassifier(**paras)

        else:
            clf_skl = tree.DecisionTreeRegressor(**paras)
            
        clf_skl.fit(X_train,y_train)  

          
        self.paras = paras
        self.clf_gbp = clf
        self.clf_skl = clf_skl

        dot_data0 = tree_prune.export_graphviz(clf,
                                               out_file=None,
                                               feature_names=self.feature_col)

        dot_data1 = export_graphviz(clf_skl,
                                   out_file=None,
                                   feature_names=self.feature_col)
        
        ''' PLOT:
        graph = pydotplus.graph_from_dot_data(dot_data0)
        Image(graph.create_png())
        '''
        
        if self.dtree_gbp:
            ls = dot_data0.split('\n')
        else:
            ls = dot_data1.split('\n')

        rules = []
        for s in ls:
            if re.findall(r"label=\"(.+?)\\",s):
                rules.append(re.findall(r"label=\"(.+?)\\",s)[0])    
        
        rules = [x for x in rules if x.split(' ')[0] != 'gini']
        
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
                print('Warning: less should be less than or equal 0')   
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



    def LeafNodesFillNaN(self, dfall:pd.DataFrame, OneGroupDfIndex:pd.Index, LeafNode):
        '''
        Filling  miss values in one seperated group:dfonegroup,
        
        which came from dfall sliced by OneGroupDfIndex
        
        for object columns, use most_freq value,
        for int and float columns, use mean value
        
        '''
        dfonegroup = dfall.loc[OneGroupDfIndex]
        
        
        df_obj = dfonegroup.blocks.get('object')
        
        #self.test = df_obj
        
        if df_obj is not None:
        
            bool_in = dfonegroup.columns.isin(df_obj.columns)
            df_numer = dfonegroup[dfonegroup.columns[~bool_in]] # ~ means logical negative
            
            df_numer_record =  df_numer.describe().T['mean'].to_frame(name=LeafNode)
            df_obj_record = df_obj.describe().T.top.to_frame(name=LeafNode)
            
            df_record = df_numer_record.append(df_obj_record)
        
        else:
            df_record = dfonegroup.describe().T['mean'].to_frame(name=LeafNode)
            
        #df_record.isnull().sum().sum()
    
        return df_record


    def OneGroupFill(self, df, df_record, OneGroupIndex, OneGroupNode):
        '''
        Inputs:
        ----
          df: DataFrame needs to be fill NaN
          
          df_record: DataFrame for LeafNode's record to fill NaN, columns are Nodes(each column is One Group)
          
          OneGroupIndex: One Group's index
          
          OneGroupNode: One Group belongs which Node
    
        '''
        dfonegroup = df.loc[OneGroupIndex]
        
        FillDict = df_record[OneGroupNode].to_dict()
        
        dfilled = dfonegroup.fillna(FillDict)
    
        return dfilled    


    def MultiGroupRun(self, fuction, task_zip,n_jobs=6):

        #Multi-process Run

        pool = ThreadPool(processes=n_jobs)
        #pool.apply_async(fuction, task_zip)
        results  = pool.starmap(fuction, task_zip)

        pool.close()
        pool.join()
        
            
        return results     

    
    
    def fit(self, df:pd.DataFrame, target_col):
        """Fit the imputer on df.

        Parameters
        ----------
        df : DataFrame.
        target_col: target_cols
        

        Returns
        -------
        self : Imputer
            Returns self.
        """

        ##init the dataset
        df_train_new = self.fuc_prep_train_data(df, target_col)
        df_train_old = self.original_df_train

        self.prep_df_train = df_train_new
        
        #df_test_new = self.load_test_data_prep()        
        #df_test_old = self.original_df_test
        
        dff = df_train_old.copy()
        
        
        if self.strategy == 'dt':
            
            #get rules list 
            rules = self.main_dt_rules(self.X_train,self.y_train)
            
            #diffrent in train and test because of index
            dfrules = self.main_dt_dfrules(rules, df_train_new) 
            
            self.rules = rules
            self.dfrules = dfrules
            
            #fill the missing values until No NaN
            GroupDict = self.main_dt_group_groupby(dfrules, 0)
        
            self.GroupDict = GroupDict
            
            train_task_zip = zip(repeat(dff),GroupDict.values(),GroupDict.keys())
            
            df_record_list = self.MultiGroupRun(self.LeafNodesFillNaN, train_task_zip,n_jobs=self.n_jobs)
            
            df_record = pd.concat(df_record_list,axis=1) 
            
            
            print(0, df_record.isnull().sum().sum())
            
            nodes_flag=-1
            while df_record.isnull().sum().sum():
                
                
                _GroupDict = self.main_dt_group_groupby(dfrules, nodes_flag)
                _train_task_zip = zip(repeat(dff),_GroupDict.values(),_GroupDict.keys())
                
                _df_record_list = self.MultiGroupRun(self.LeafNodesFillNaN, _train_task_zip,n_jobs=self.n_jobs)
                
                _df_record = pd.concat(_df_record_list,axis=1)  
                    
                FillDict=dict() #build a new dictionary to fill nan
                for k,v in df_record.to_dict().items():
                    
                    if len(k[:nodes_flag]) == 1:
                        FillDict[k] =  _df_record.to_dict()[k[:nodes_flag][0]]
                    else:
                        FillDict[k] =  _df_record.to_dict()[k[:nodes_flag]]
                    
                df_record = df_record.fillna(FillDict)
 
                print(nodes_flag, df_record.isnull().sum().sum())
                
                nodes_flag = nodes_flag-1
                
                if nodes_flag == -len(rules):
                    break
                
            self.records = df_record
          


        #method for hireachical tree    
        elif self.strategy == 'ht':

            df_prep = df_train_new.copy()
              
            #for feature methods:
            if self.htree_features == 'tree' :
                
                tree_ims = TreeImpFeat(self.X_train,self.y_train, feature_names=self.feature_col)
                sel_cols = tree_ims[tree_ims > self.htree_features_alpha].index
                df_prep = df_prep[sel_cols]
                
                
            elif self.htree_features == 'prune_tree':
                rules = self.main_dt_rules(self.X_train,self.y_train)
                self.rules = rules
                sel_cols = [x.split(' <= ')[0] for x in rules]
                
                df_prep = df_prep[list(set(sel_cols))]
                
            else:
                df_prep = df_prep
                
            self.df_prep_ht = df_prep
            D, Z  = self.main_ht_get_Z(df_prep)
            
            self.D = D            
            self.Z = Z
            
            
            dmax = Z[:,2].max()
            #dmin = Z[:,2].min()
            
            dis_start = self.htree_dis_start 
            
            dis_step = self.htree_dis_step
            
            clusters = self.main_ht_get_cluster(Z, dis_start)
            
            print('Number of classes are:', np.unique(clusters).shape[0])
            GroupDict = pd.DataFrame(clusters, index= dff.index).groupby(0).groups
            
            self.GroupDict = GroupDict
            
            
            task_zip = zip(repeat(dff),GroupDict.values(),GroupDict.keys())
            
            df_record_list = self.MultiGroupRun(self.LeafNodesFillNaN,task_zip,n_jobs=self.n_jobs)
            
            df_record = pd.concat(df_record_list,axis=1) 
            
            print(dis_start, df_record.isnull().sum().sum()) 
            
            self.main_ht_plot_cluster(Z, 80, figname='ht_80.png')               
            #self.main_ht_plot_cluster(Z, 50, figname='ht_50.png')
            #self.main_ht_plot_cluster(Z, 30, figname='ht_30.png')               
            #self.main_ht_plot_cluster(Z, 10, figname='ht_10.png')   
            
            while df_record.isnull().sum().sum():
                
                dis_start = dis_start + dis_step
                
                _clusters = self.main_ht_get_cluster(Z, dis_start)
                
                print('Number of classes are:', np.unique(_clusters).shape[0])
                
                _GroupDict = pd.DataFrame(_clusters, index= dff.index).groupby(0).groups
                
                _train_task_zip = zip(repeat(dff),_GroupDict.values(),_GroupDict.keys())
                
                _df_record_list = self.MultiGroupRun(self.LeafNodesFillNaN, _train_task_zip,n_jobs=self.n_jobs)
                
                _df_record = pd.concat(_df_record_list,axis=1)   
                
                FillDict=dict() #build a new dictionary to fill nan
                for k,v in GroupDict.items():
                    for _k, _v in _GroupDict.items():
                        if v[0] in _v:
                            FillDict[k]=_df_record[_k].to_dict()
                   
                df_record = df_record.fillna(FillDict)
 
                print(dis_start, df_record.isnull().sum().sum())                
                
                if dis_start > dmax:
                    break
            
            
            self.records = df_record

        else:
            print('strategy does not support, \'dt\',\'ht\' can be used' )
            self.records = pd.DataFrame([],index=self.variable_col)
            
        return self
        
           
    def transform(self, df):
        
        """Impute all missing values in df.

        Parameters
        ----------
        df : DataFrame.
        """
        
        
        df_test_new = self.fuc_prep_test_data(df)      
        df_test_old = self.original_df_test    
        
        self.prep_df_test = df_test_new
        
        dff_t = df_test_old.copy()
        
        #check if train is equal to test    
        if self.original_df_test[self.variable_col].equals(self.original_df_train[self.variable_col]):
            print('transfrom data has an equal array with fit data')
            
            #self.df_record.to_dict()
            self.GroupDict_test = self.GroupDict
                
            task_zip = zip(repeat(dff_t),
                           repeat(self.records),
                           self.GroupDict_test.values(),
                           self.GroupDict_test.keys())
            
            df_fill_list = self.MultiGroupRun(self.OneGroupFill,task_zip,n_jobs=self.n_jobs)            
            df_fill = pd.concat(df_fill_list) 
            
            df_test_filled= pd.DataFrame([],index=dff_t.index).join(df_fill)
            
        
        else :
            if self.strategy == 'dt':
                
                test_dfrules = self.main_dt_dfrules(self.rules, df_test_new)
                
                self.GroupDict_test = self.main_dt_group_groupby(test_dfrules, 0)
                
                d=  pairwise_distances(
                np.array(list(self.GroupDict.keys())),
                np.array(list(self.GroupDict_test.keys())),
                metric='rogerstanimoto'
                )
                
                dfd = pd.DataFrame(d,index=list(self.GroupDict.keys()),
                             columns=list(self.GroupDict_test.keys()))
                
                tstrd = dfd.idxmin().to_dict() #train, test (True,False) map dict
                
                GroupDict_test2=dict()
                for k,v in self.GroupDict_test.items():
                    GroupDict_test2[tstrd[k]]=v
                    
                self.GroupDict_test2 = GroupDict_test2
                
                task_zip = zip(repeat(dff_t),
                               repeat(self.records),
                               self.GroupDict_test2.values(),
                               self.GroupDict_test2.keys())
                
                df_fill_list = self.MultiGroupRun(self.OneGroupFill,task_zip,n_jobs=self.n_jobs)            
                df_fill = pd.concat(df_fill_list) 
                
                df_test_filled= pd.DataFrame([],index=dff_t.index).join(df_fill)
                    
            
            #method for hireachical tree    
            elif self.strategy == 'ht':
                
                dfall = self.prep_df_train.append(self.prep_df_test)
                
                D, _  = self.main_ht_get_Z(dfall)
                
                #convert to distence matrixs
                df_dis = pd.DataFrame(squareform(D),
                                      index=dfall.index,
                                      columns=dfall.index)

                #each row is test sample, each column is train sample
                D_test_train = df_dis.loc[self.prep_df_test.index][self.prep_df_train.index]
            
                #get distence for each test sample to trains's Groups            
                otg=[]
                for k,v in self.GroupDict.items():
                    otg.append(D_test_train[v].mean(axis=1).to_frame(name=k))
                df_group_ = pd.concat(otg,axis=1)
                
                #Get the test's groups's belongs by min distence to the Train Groups
                self.GroupDict_test = df_group_.idxmin(axis=1).to_frame(name='Nodes').groupby('Nodes').groups
                task_zip = zip(repeat(dff_t),
                               repeat(self.records),
                               self.GroupDict_test.values(),
                               self.GroupDict_test.keys())
                
                df_fill_list = self.MultiGroupRun(self.OneGroupFill,task_zip,n_jobs=self.n_jobs)            
                df_fill = pd.concat(df_fill_list) 
                
                df_test_filled= pd.DataFrame([],index=dff_t.index).join(df_fill)
                
                
                
            else:
                print('strategy does not support, \'dt\',\'ht\' can be used' )
                df_test_filled = pd.DataFrame()
            
        return df_test_filled




#for k,v in d0.items():
    
