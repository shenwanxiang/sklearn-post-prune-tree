新方法DTGetBestParas和sklearn的使用类似，具体方法如下：
>>> from sklearn.datasets import load_boston

>>> import DTGetBestParas

>>> boston = load_boston()

>>> X = boston.data

>>> y = boston.target
    

#类的初始化
>>> op = DTGetBestParas(method='prune')
    
#类的fit，传入的X,y为数组或者dfx,dfy
>>> op.fit(X,y)

#获取最佳参数
>>> print(op.max_depth,
          op.min_samples_leaf,
          op.min_samples_split,
          op.max_leaf_nodes)
