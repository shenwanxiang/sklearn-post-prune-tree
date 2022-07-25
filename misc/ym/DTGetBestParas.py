#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 14:26:33 2017

#################################################
Getting best parasmeter for descion tree models,
including the most important four parameters
max_depth,
min_samples_leaf,
min_samples_split,
max_leaf_nodes
#################################################

@author: charleshen



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import tree
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from collections import Iterable
from sklearn.metrics import matthews_corrcoef, make_scorer,roc_auc_score,mean_squared_error



#import post prune tree method
import tree_prune

from copy import deepcopy
from IPython.display import Image
from operator import itemgetter
from time import time
import pydotplus


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class GBP(object):

    def __init__(self,method = 'prune',n_iter = 6,
                 cv = 5,n_jobs = 4,random_state = 100,
                 prune_alpha = 0.5,both_alpha = 0.85,
                 seed_list = list(range(0,500,50)),
                 max_leaves = 250,
                 test_size=0.1,
                 figtitle = 'dataset_test',
                 **kwargs):
        
        #set paras to run
        self.method = method
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.prune_alpha = prune_alpha
        self.both_alpha = both_alpha
        self.max_leaves = max_leaves
        self.seed_list = seed_list
        self.figtitle = figtitle + '_' + method
        self.test_size = test_size
        
        #optimal paras to return
        self.max_depth = None
        self.min_samples_leaf = None
        self.min_samples_split = None
        self.max_leaf_nodes = None
        self.best_score = None
        self.test_roc_mse = None
        



    def my_auc(self, estimator, X, y):
        y_score = estimator.predict_proba(X)[:,1]  # You could also use the binary predict, but probabilities should give you a more realistic score.
        return roc_auc_score(y, y_score)



    #regression score
    def my_nmse(self, estimator, X, y):
        y_pre = estimator.predict(X)  # You could also use the binary predict, but probabilities should give you a more realistic score.
        return -mean_squared_error(y, y_pre)



        
            
    def get_min_samples(self, N):
        '''
        N ---   number of the samples
        return
                number of min_samples_split , min_samples_leaf
        '''
        if N>=1000000000:
            return 500, 250
        else:
            if N >=1000000:
                s = min(0.0000002*N+200,400)
                return int(s),int(s/2)
            else:
                if N>=10000:
                    s = min(0.00015*N+50,200)
                    return int(s),int(s/2)
                else:
                    return 50,25



    def report(self,grid_scores, n_top=3):
        '''
        Report top n_top parameters settings, default n_top=3.
    
        Args
        ----
        grid_scores -- output from grid or random search
        n_top -- how many to report, of top models
    
        Returns
        -------
        top_params -- [dict] top parameter settings found in
                      search
        '''
        top_scores = sorted(grid_scores,
                            key=itemgetter(1),
                            reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print(("Mean validation score: "
                   "{0:.3f} (std: {1:.3f})").format(
                   score.mean_validation_score,
                   np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
    
        return top_scores[0].parameters
    
    
    def run_gridsearch(self, X, y, clf, param_grid, cv=5, n_jobs = 1,
                       scoring = make_scorer(matthews_corrcoef)):
        '''
        Run a grid search for best Decision Tree parameters.
    
        Args
        ----
        X -- features
        y -- targets (classes)
        cf -- scikit-learn Decision Tree
        param_grid -- [dict] parameter settings to test
        cv -- fold of cross-validation, default 5
    
        Returns
        -------
        top_params -- [dict] from report()
        '''
        grid_search = GridSearchCV(clf,
                                   param_grid=param_grid,
                                   scoring = scoring,
                                   cv=cv,n_jobs = n_jobs)
        start = time()
        grid_search.fit(X, y)
        
        best_score = grid_search.best_score_
        
        print(("\nGridSearchCV took {:.2f} "
               "seconds for {:d} candidate "
               "parameter settings.").format(time() - start,
                    len(grid_search.grid_scores_)))
    
        top_params = self.report(grid_search.grid_scores_, 3)
        
        top_params['best_score'] = best_score
        return  top_params



    def tune_paras(self, clf,X,y, random_state = 100,cv =5,scoring = 'roc_auc',n_jobs =1):
        
        min_samples_split,min_samples_leaf = self.get_min_samples(len(y))
        clf.min_samples_split = int(0.5*min_samples_split)
        clf.min_samples_leaf = int(0.5*min_samples_leaf)
        clf.fit(X,y)        
        depth = tree_prune.get_max_depth(clf)
        
        if depth > 10:
            depth_ls = list(range(6,depth,2))
        else:
            depth_ls = [4,6,8,10,12]
            
        #depth_ls.append(None)
        
        #tune dist
        param_dist = {'max_depth': depth_ls}

        clf.random_state = random_state
        clf.fit(X,y)
        ts_gs = self.run_gridsearch(X, y, clf, 
                               param_grid =param_dist,
                               cv=cv, scoring = scoring,
                               n_jobs = n_jobs)
        print('*************',ts_gs)
        ts_gs['min_samples_split'] = clf.min_samples_split
        ts_gs['min_samples_leaf'] = clf.min_samples_leaf
        ts_gs['max_leaf_nodes'] = tree_prune.get_n_leaves(clf)

        return ts_gs



    def get_optimal_leaves(self, clf,X_train,y_train,n_iterations=5,random_state = 100,alpha = 0.5, max_leaves = 500,test_size = 0.1,n_jobs=-1):
        
        leaves = tree_prune.get_n_leaves(clf)
        
        
        if leaves > max_leaves:
            mycut = max_leaves
        else:
            mycut = leaves
            
            
            
            
            
        dfscores = tree_prune.prune_path(clf, 
                                         X_train,
                                         y_train,
                                         max_n_leaves=mycut, 
                                         n_iter=n_iterations,
                                         random_state=random_state,
                                         test_size=test_size,
                                         n_jobs = n_jobs)    
    
        dfmeans = dfscores.mean(axis=1)
        dfstds = dfscores.std(axis=1)
        
        
        cutid = dfmeans.idxmax()
        
        
        return cutid, dfmeans, dfstds


    def plot_prune(self, dfmeans,dfstds,title='Prune_performance'):
        
        plt.figure(figsize=(16,8))
        import matplotlib 
        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20) 
        plt.plot(dfmeans.index, dfmeans.values,lw = 5)
        plt.plot([dfmeans.idxmax(),dfmeans.idxmax()], [dfmeans.min()-dfstds.min(),dfmeans.max()+dfstds.max()],'--', lw=4, c='r',label='Prune')        
        
        plt.text(dfmeans.idxmax(),dfmeans.mean(),str(dfmeans.idxmax()),  fontsize = 24)
        
        plt.plot(dfmeans.index, dfmeans.values + 1.2 * dfstds.values, lw=4, c='0.7')
        plt.plot(dfmeans.index, dfmeans.values - 1.2 * dfstds.values, lw=4, c='0.7')
        plt.xlabel('Number of leaf nodes',fontsize = 24)
        plt.ylabel('Cross validated score',fontsize = 24)
        plt.legend(fontsize = 24)
        plt.title(title,fontsize=26)
        plt.savefig(title+'.png')
        plt.show()

     

    def fit(self, X,y,X_test = None,y_test = None):
        '''
        input:
            X: numpy array
            y: numpy array
            
        return:
            paras: dict
            
            clf: tree object
            
        '''
        
        #X = df[df.columns[:-1]].values
        #y = df[df.columns[-1]].values
        

            
        if len(np.unique(y)) == 2:
            clf = tree.DecisionTreeClassifier()  
            clf_prune = tree_prune.DecisionTreeClassifier()
            
            scoring = self.my_auc #auc score
            self.cv_score_name = 'AUC'
            
        else:
            clf = tree.DecisionTreeRegressor() 
            clf_prune = tree_prune.DecisionTreeRegressor() 
            scoring = self.my_nmse #negative mean squared error
            self.cv_score_name = 'NMSE'            

            
        clf.random_state = self.random_state
        clf_prune.random_state = self.random_state
        
        
        clf = clf.fit(X,y)
        
        depth = tree_prune.get_max_depth(clf)
        leaf_samples,node_samples = tree_prune.get_min_sample_leaf_split(clf)
        leaf_nodes =  tree_prune.get_n_leaves(clf)
        
        if self.method == 'none':
            paras = {}
            paras['max_depth'] = depth
            paras['min_samples_split'] = node_samples
            paras['min_samples_leaf'] = leaf_samples
            paras['max_leaf_nodes'] = leaf_nodes
            paras['best_score'] = None
            clf = deepcopy(clf)
            #self.roc_mse = scoring(clf,X_test, y_test)   
            
        elif self.method == 'cal': 
            paras = {}
            min_samples_split,min_samples_leaf = self.get_min_samples(len(y))
            clf.min_samples_split = min_samples_split
            clf.min_samples_leaf = min_samples_leaf
            clf2 = clf.fit(X,y)
           
            clf = deepcopy(clf2)
            #self.roc_mse = scoring(clf2,X_test, y_test) 
            
            depth = tree_prune.get_max_depth(clf)
            leaf_samples,node_samples = tree_prune.get_min_sample_leaf_split(clf)       
            paras['max_depth'] = depth
            paras['min_samples_split'] = node_samples
            paras['min_samples_leaf'] = leaf_samples
            paras['max_leaf_nodes'] =  tree_prune.get_n_leaves(clf)
            paras['best_score'] = None  
            
        elif self.method == 'tune':
            paras = self.tune_paras(clf,X,y, 
                               random_state = self.random_state,
                               cv =self.cv, 
                               scoring = scoring,
                               n_jobs =self.n_jobs)
            #clf = clf(
            #
            
        elif self.method == 'prune':
            #clf_prune.max_depth = depth
            #clf_prune.min_samples_split = node_samples
            #clf_prune.min_samples_leaf = leaf_samples
    
    
            if depth >= 25:
                clf_prune.max_depth = 25
            else:
                clf_prune.max_depth = depth        
            
            min_samples_split,min_samples_leaf = self.get_min_samples(len(y))
            clf_prune.min_samples_split = int(0.8*min_samples_split)
            clf_prune.min_samples_leaf = int(0.8*min_samples_leaf)
            
            clf_prune.fit(X,y)
            
            clf1 = deepcopy(clf_prune)
    
            print('get optimal n_leaves,max leaves of the tree is %d......\n' % tree_prune.get_n_leaves(clf1))  
            
            n_leaves,dfmeans,dfstds = self.get_optimal_leaves(clf1,
                                                            X,
                                                            y,
                                                            n_iterations=self.n_iter,
                                                            random_state = self.random_state, 
                                                            alpha = self.prune_alpha,
                                                            max_leaves = self.max_leaves,
                                                            test_size= self.test_size,
                                                            n_jobs = self.n_jobs)
            
            print('optimal n_leaves is: %d, begin pruning tree...\n' % n_leaves)
            #prune
            clf_prune = clf_prune.prune(n_leaves)
            
            print('pruning is finished')
            
            #get pruned tree's best parameters
            max_depth = tree_prune.get_max_depth(clf_prune)
            min_sample_leaf, min_sample_split = tree_prune.get_min_sample_leaf_split(clf_prune)
            
            paras = {}
            paras['max_depth'] = max_depth
            paras['min_samples_split'] = min_sample_split
            paras['min_samples_leaf'] = min_sample_leaf      
            paras['max_leaf_nodes'] =  n_leaves
            paras['best_score'] = dfmeans.max()
            
            self.plot_prune(dfmeans,dfstds,title=self.figtitle)
            
        elif self.method == 'both':
            
            print('tuning,please wait...\n')
            tune_dict = self.tune_paras(clf,X,y, 
                                   random_state = self.random_state,
                                   cv =self.cv, 
                                   scoring = scoring,
                                   n_jobs =self.n_jobs)
        
            clf_prune.max_depth = tune_dict['max_depth']
            clf_prune.min_samples_split = tune_dict['min_samples_split']
            clf_prune.min_samples_leaf = tune_dict['min_samples_leaf']
            
            
            clf_prune.fit(X,y)
    
    
            clf2= deepcopy(clf_prune)
            
            print('get optimal n_leaves...,max leaves of the tree is %d\n' % tree_prune.get_n_leaves(clf2))          
            
            #find best prune parameter: n_leaves
            n_leaves,dfmeans,dfstds = self.get_optimal_leaves(clf2,
                                                            X,
                                                            y,
                                                            n_iterations=self.n_iter,
                                                            random_state = self.random_state, 
                                                            alpha = self.both_alpha,
                                                            max_leaves = self.max_leaves,
                                                            test_size=self.test_size,
                                                            n_jobs = self.n_jobs)
    
            print('optimal n_leaves is: %d, begin pruning tree...\n' % n_leaves)
            
            #prune
            clf_prune = clf_prune.prune(n_leaves)
            
            print('pruning is finished') 
            
            #get pruned tree's best parameters
            max_depth = tree_prune.get_max_depth(clf_prune)
            min_sample_leaf, min_sample_split = tree_prune.get_min_sample_leaf_split(clf_prune)
            
            paras = {}
            paras['max_depth'] = max_depth
            paras['min_samples_split'] = min_sample_split
            paras['min_samples_leaf'] = min_sample_leaf
            paras['max_leaf_nodes'] =  n_leaves
            paras['best_score'] = dfmeans.max()
            
            self.plot_prune(dfmeans,dfstds,title=self.figtitle)
            
        else:
            print("get empty parameters dict, only 'none', 'cal', 'tune', 'prune' and 'both' are supported in the method currently")
            paras = {} 
            paras['max_depth'] = None
            paras['min_samples_split'] = None
            paras['min_samples_leaf'] = None
            paras['max_leaf_nodes'] = None
            paras['best_score'] = None  
         
            
        if self.method == 'both':
            self.clf = deepcopy(clf_prune)
        elif self.method == 'prune':
            self.clf = deepcopy(clf_prune)
        else:
            self.clf = deepcopy(clf)

        
        if X_test is not None:
            self.test_roc_mse = scoring(clf,X_test, y_test)
        else:
            self.test_roc_mse = None
        
        self.max_depth = paras['max_depth']
        self.min_samples_split = paras['min_samples_split']
        self.min_samples_leaf = paras['min_samples_leaf']  
        self.max_leaf_nodes = paras['max_leaf_nodes']
        self.best_score = paras['best_score']
        return self

    
    def TreeImpFeat(self, X,y, **kwargs):
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
              
            for i in self.seed_list:
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
              
            for i in self.seed_list:
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
        #return dfall

    def plot_tree(self, clf, feature_names):
        '''
        input:
            tree of clf without pruning
            feature_importance    
        output: 
            
        '''
        dot_data = export_graphviz(clf, out_file=None,
                                   feature_names=feature_names, 
                                   filled=True, rounded=True, 
                                   special_characters=True)
        
        graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
        Image(graph.create_png())
            
 
if __name__ == "__main__":
    '''
    Test for DTGetBestParas Class
    '''
    print(__doc__)
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    #init
    op = GBP(method='prune',figtitle='boston')
    
    #fit
    op.fit(X,y)
    #print(op.clf)
    #get result
    print(op.max_depth,
          op.min_samples_leaf,
          op.min_samples_split,
          op.max_leaf_nodes,
          op.test_roc_mse)




    