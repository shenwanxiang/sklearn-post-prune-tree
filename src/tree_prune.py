#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 19 11:01:33 2017

@author: charleshen

This module gathers prune tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
Created on Fri Aug 18 09:31:08 2017

Prunes the tree to obtain the optimal subtree with n_leaves leaves.

more information please see:

[1] J. Friedman and T. Hastie, "The elements of statistical learning", 2001, section 9.2.1

Code is originally adapted from MILK: Machine Learning Toolkit(sklearn-0.18.0)

"""



from __future__ import division

import numbers
import numpy as np
from scipy.sparse import issparse
from abc import ABCMeta, abstractmethod
from warnings import warn

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.externals import six
from sklearn.externals.six.moves import xrange
#from ..feature_selection.selector_mixin import SelectorMixin
#from sklearn.feature_selection.from_model import _LearntSelectorMixin
#from sklearn.feature_selection.base import SelectorMixin


from sklearn.utils import check_array, check_random_state
#from ..utils.validation import check_arrays

from base_tree import _tree_prune as _tree








__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CLASSIFICATION = {
    "gini": _tree.Gini,
    "entropy": _tree.Entropy,
}

REGRESSION = {
    "mse": _tree.MSE,
}






def pruning_order(self, max_to_prune=None):
    """Compute the order for which the tree should be pruned.

    The algorithm used is weakest link pruning. It removes first the nodes
    that improve the tree the least.


    Parameters
    ----------
    max_to_prune : int, optional (default=all the nodes)
        maximum number of nodes to prune

    Returns
    -------
    nodes : numpy array
        list of the nodes to remove to get to the optimal subtree.

    References
    ----------

    .. [1] J. Friedman and T. Hastie, "The elements of statistical
    learning", 2001, section 9.2.1

    """

    def _get_terminal_nodes(children):
        """Lists the nodes that only have leaves as children"""
        leaves = np.where(children[:,0]==_tree.TREE_LEAF)[0]
        child_is_leaf = np.in1d(children, leaves).reshape(children.shape)
        return np.where(np.all(child_is_leaf, axis=1))[0]

    def _next_to_prune(tree, children=None):
        """Weakest link pruning for the subtree defined by children"""

        if children is None:
            children = tree.children

        t_nodes = _get_terminal_nodes(children)
        g_i = tree.init_error[t_nodes] - tree.best_error[t_nodes]

        return t_nodes[np.argmin(g_i)]

    if max_to_prune is None:
        max_to_prune = self.node_count - sum(self.children_left == _tree.TREE_UNDEFINED)

    children = np.array([self.children_left.copy(), self.children_right.copy()]).T
    nodes = list()

    while True:
        node = _next_to_prune(self, children)
        nodes.append(node)

        if (len(nodes) == max_to_prune) or (node == 0):
            return np.array(nodes)

        #Remove the subtree from the children array
        children[children[node], :] = _tree.TREE_UNDEFINED
        children[node, :] = _tree.TREE_LEAF

def prune(self, n_leaves):
    """Prunes the tree to obtain the optimal subtree with n_leaves leaves.


    Parameters
    ----------
    n_leaves : int
        The final number of leaves the algorithm should bring

    Returns
    -------
    tree : a Tree object
        returns a new, pruned, tree

    References
    ----------

    .. [1] J. Friedman and T. Hastie, "The elements of statistical
    learning", 2001, section 9.2.1

    """
    true_node_count = self.node_count - sum(self.children_left == _tree.TREE_UNDEFINED)
    leaves = np.where(self.children_left == _tree.TREE_LEAF)[0]
    to_remove_count = true_node_count - 2*n_leaves + 1

    nodes_to_remove = pruning_order(self, max_to_prune = to_remove_count/2)

    # self._copy is gone, but this does the same thing
    out_tree = _tree.Tree(*self.__reduce__()[1])
    out_tree.__setstate__(self.__getstate__().copy())

    for node in nodes_to_remove:
        #TODO: Add a Tree method to remove a branch of a tree
        out_tree.children_left[out_tree.children_left[node]] = _tree.TREE_UNDEFINED
        out_tree.children_right[out_tree.children_left[node]] = _tree.TREE_UNDEFINED
        out_tree.children_left[out_tree.children_right[node]] = _tree.TREE_UNDEFINED
        out_tree.children_right[out_tree.children_right[node]] = _tree.TREE_UNDEFINED
        out_tree.children_left[node] = _tree.TREE_LEAF
        out_tree.children_right[node] = _tree.TREE_LEAF

    # FIXME: currently should not change node_count, after deletion
    # this is not number of nodes in the tree
    #out_tree.node_count -= 2*len(nodes_to_remove)

    return out_tree


class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_density,
                 max_features,
                 compute_importances,
                 random_state):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features

        if compute_importances:
            pass

        self.compute_importances = compute_importances
        self.random_state = random_state

        self.n_features_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.find_split_ = _tree.TREE_SPLIT_BEST

        self.tree_ = None



    # TODO: update this method if necessary after updating Tree class prune method
    def prune(self, n_leaves):
        """Prunes the decision tree

        This method is necessary to avoid overfitting tree models. While broad
        decision trees should be computed in the first place, pruning them
        allows for smaller trees.

        Parameters
        ----------
        n_leaves : int
            the number of leaves of the pruned tree

        """
        self.tree_ = prune(self.tree_, n_leaves)
        return self

    def fit(self, X, y,
            sample_mask=None, X_argsorted=None,
            check_input=True, sample_weight=None):
        """Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. Use ``dtype=np.float32``
            and ``order='F'`` for maximum efficiency.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).
            Use ``dtype=np.float64`` and ``order='C'`` for maximum
            efficiency.

        sample_mask : array-like, shape = [n_samples], dtype = bool or None
            A bit mask that encodes the rows of ``X`` that should be
            used to build the decision tree. It can be used for bagging
            without the need to create of copy of ``X``.
            If None a mask will be created that includes all samples.

        X_argsorted : array-like, shape = [n_samples, n_features] or None
            Each column of ``X_argsorted`` holds the row indices of ``X``
            sorted according to the value of the corresponding feature
            in ascending order.
            I.e. ``X[X_argsorted[i, k], k] <= X[X_argsorted[j, k], k]``
            for each j > i.
            If None, ``X_argsorted`` is computed internally.
            The argument is supported to enable multiple decision trees
            to share the data structure and to avoid re-computation in
            tree ensembles. For maximum efficiency use dtype np.int32.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
        """

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=None)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            for k in xrange(self.n_outputs_):
                unique = np.unique(y[:, k])
                self.classes_.append(unique)
                self.n_classes_.append(unique.shape[0])
                y[:, k] = np.searchsorted(unique, y[:, k])

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        if is_classification:
            criterion = CLASSIFICATION[self.criterion](self.n_outputs_,
                                                       self.n_classes_)
        else:
            criterion = REGRESSION[self.criterion](self.n_outputs_)

        '''                    
        # Convert data
        if (getattr(X, "dtype", None) != DTYPE or
                X.ndim != 2 or
                not X.flags.fortran):
            X = check_array(X, dtype=DTYPE, order="F")
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        '''

        # Check parameters
        max_depth = np.inf if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if self.min_samples_split <= 0:
            raise ValueError("min_samples_split must be greater than zero.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero.")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if self.min_density < 0.0 or self.min_density > 1.0:
            raise ValueError("min_density must be in [0, 1]")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if sample_mask is not None:
            sample_mask = np.asarray(sample_mask, dtype=np.bool)

            if sample_mask.shape[0] != n_samples:
                raise ValueError("Length of sample_mask=%d does not match "
                                 "number of samples=%d"
                                 % (sample_mask.shape[0], n_samples))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if X_argsorted is not None:
            X_argsorted = np.asarray(X_argsorted, dtype=np.int32,
                                     order='F')
            if X_argsorted.shape != X.shape:
                raise ValueError("Shape of X_argsorted does not match "
                                 "the shape of X")

        # Set min_samples_split sensibly
        min_samples_split = max(self.min_samples_split,
                                2 * self.min_samples_leaf)

        # Build tree
        self.tree_ = _tree.Tree(self.n_features_, self.n_classes_,
                                self.n_outputs_, criterion, max_depth,
                                min_samples_split, self.min_samples_leaf,
                                self.min_density, max_features,
                                self.find_split_, random_state)


                  
        # Convert data
        if (getattr(X, "dtype", None) != DTYPE or
                X.ndim != 2 or
                not X.flags.fortran):
            X = check_array(X, dtype=DTYPE, order="F")
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)


        self.tree_.build(X, y,
                         sample_weight=sample_weight,
                         sample_mask=sample_mask,
                         X_argsorted=X_argsorted)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = check_array(X, dtype=DTYPE, order="F")

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        proba = self.tree_.predict(X)

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba[:, 0], axis=1),
                                          axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in xrange(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0, 0]

            else:
                return proba[:, :, 0]

##########################new#################################
    def apply(self, X, check_input=True):
        from sklearn.utils.validation import check_is_fitted
        """
        Returns the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, 'tree_')
        #X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree
        .. versionadded:: 0.18
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)


    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance [4]_.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.tree_ is None:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        return self.tree_.compute_feature_importances()


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """A decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    `classes_` : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    `n_classes_` : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    `feature_importances_` : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """
    def __init__(self,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.0,
                 max_features=None,
                 compute_importances=False,
                 random_state=None):
        super(DecisionTreeClassifier, self).__init__(criterion,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf,
                                                     min_density,
                                                     max_features,
                                                     compute_importances,
                                                     random_state)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by arithmetical order.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = check_array(X, dtype=DTYPE, order="F")

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, 0, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in xrange(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in xrange(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A tree regressor.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. The only supported
        criterion is "mse" for the mean squared error.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=n_features`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    `feature_importances_` : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized)total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)

    R2 scores (a.k.a. coefficient of determination) over 10-folds CV:

    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    """
    def __init__(self,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.0,
                 max_features=None,
                 compute_importances=False,
                 random_state=None):
        super(DecisionTreeRegressor, self).__init__(criterion,
                                                    max_depth,
                                                    min_samples_split,
                                                    min_samples_leaf,
                                                    min_density,
                                                    max_features,
                                                    compute_importances,
                                                    random_state)


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    See also
    --------
    ExtraTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """
    def __init__(self,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.0,
                 max_features="auto",
                 compute_importances=False,
                 random_state=None):
        super(ExtraTreeClassifier, self).__init__(criterion,
                                                  max_depth,
                                                  min_samples_split,
                                                  min_samples_leaf,
                                                  min_density,
                                                  max_features,
                                                  compute_importances,
                                                  random_state)

        self.find_split_ = _tree.TREE_SPLIT_RANDOM


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    See also
    --------
    ExtraTreeClassifier : A classifier base on extremely randomized trees
    sklearn.ensemble.ExtraTreesClassifier : An ensemble of extra-trees for
        classification
    sklearn.ensemble.ExtraTreesRegressor : An ensemble of extra-trees for
        regression

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """
    def __init__(self,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.0,
                 max_features="auto",
                 compute_importances=False,
                 random_state=None):
        super(ExtraTreeRegressor, self).__init__(criterion,
                                                 max_depth,
                                                 min_samples_split,
                                                 min_samples_leaf,
                                                 min_density,
                                                 max_features,
                                                 compute_importances,
                                                 random_state)

        self.find_split_ = _tree.TREE_SPLIT_RANDOM



def prune_path(clf, X, y, max_n_leaves=10, n_iter=10,
                          test_size=0.1, random_state=None, n_jobs=1):
    """Cross validation of scores for different values of the decision tree.

    This function allows to test what the optimal size of the post-pruned
    decision tree should be. It computes cross validated scores for different
    size of the tree.

    Parameters
    ----------
    clf: decision tree estimator object
        The object to use to fit the data

    X: array-like of shape at least 2D
        The data to fit.

    y: array-like
        The target variable to try to predict.

    max_n_leaves : int, optional (default=10)
        maximum number of leaves of the tree to prune

    n_iter : int, optional (default=10)
        Number of re-shuffling & splitting iterations.

    test_size : float (default=0.1) or int
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
        
    n_jobs : int or -1(default=1)
        -1 mean using all cpu(thread) to run, 3 means using 3 cpus.    

    Returns
    -------
    scores : list of list of floats,VIP: list may not equal lenght
        The scores of the computed cross validated trees grouped by tree size.
        scores[0] correspond to the values of trees of size max_n_leaves and
        scores[-1] to the tree with just two leaves.

    """
    

    from sklearn.base import clone
    from sklearn.cross_validation import StratifiedShuffleSplit,ShuffleSplit
    from sklearn.metrics import roc_auc_score,mean_squared_error
    from multiprocessing.dummy import Pool as ThreadPool
    from itertools import repeat
    import pandas as pd
    #import copy
    
    #classification score
    def my_auc(estimator, X, y):
        y_score = estimator.predict_proba(X)[:,1]  # You could also use the binary predict, but probabilities should give you a more realistic score.
        return roc_auc_score(y, y_score)
    
    #regression score
    def my_nmse(estimator, X, y):
        y_pre = estimator.predict(X)  # You could also use the binary predict, but probabilities should give you a more realistic score.
        return -mean_squared_error(y, y_pre)
    

    if len(np.unique(y)) == 2:        
        scoring_fuc = my_auc
        
    else:
        scoring_fuc = my_nmse
     
    def multip_run(fuction,task_zip,n_jobs = 1):

        #Multi-process Run

        pool = ThreadPool(processes=n_jobs)
        results  = pool.starmap(fuction, task_zip)
        pool.close()
        pool.join()
        return results 

    def OneFoldCut(clf,X_train, y_train,X_test,y_test,max_n_leaves):
        estimator = clone(clf)
            
        fitted = estimator.fit(X_train, y_train)
        
        if max_n_leaves < get_n_leaves(fitted):
            n_leaves = max_n_leaves
        
        else:
            n_leaves = get_n_leaves(fitted)
        
        print('###### Iters true start leaves is %d #######' % n_leaves)
                        
        #cut_num = list(range(2,n_leaves, 1))
        cut_num = list(range(n_leaves-1,1,-1))
        #n = len(cut_num)
        loc_indexs = []
        loc_scores = []
        for i in cut_num:
            #clf1 = copy.deepcopy(fitted)
            #clf1 = clone(fitted)
            #clf1.prune(i)
            fitted.prune(i)
            onescore = scoring_fuc(fitted,X_test,y_test)
            #onescore = scoring_fuc(clf1,X_test,y_test)
            loc_scores.append(onescore)
            loc_indexs.append(i)
        
        S = pd.DataFrame(loc_scores,index=loc_indexs)

        return S


    #scores = list()
    if len(np.unique(y)) == 2:            
        kf = StratifiedShuffleSplit(y,
                                    n_iter = n_iter, 
                                    test_size= test_size,
                                    random_state=random_state)
    else:
        kf = ShuffleSplit(len(y),
                        n_iter = n_iter, 
                        test_size= test_size,
                        random_state=random_state)
    
    X_trains = [X[tr] for tr,ts in kf]
    y_trains = [y[tr] for tr,ts in kf]
    
    X_tests = [X[ts] for tr,ts in kf]
    y_tests = [y[ts] for tr,ts in kf]
    
    task_zip = zip(repeat(clf),
                   X_trains,
                   y_trains,
                   X_tests,
                   y_tests,
                   repeat(max_n_leaves))
    
    scores = multip_run(OneFoldCut,task_zip,n_jobs = n_jobs)
    
    df = pd.concat(scores,axis=1)
    df.columns = range(len(df.columns))

    return df #zip(*scores)



def get_max_depth(clf):
    """
    Get the depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_max_depths(clf)
    2
    """
    tree =clf.tree_
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths)  
    return max(depths)


def get_n_leaves(clf):
    """
    Get the leaves of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_n_leaves(clf)
    4
    """
    leaves = clf.tree_.children_left == -1
    leaves = np.arange(0,clf.tree_.node_count)[leaves]
    return len(leaves)


def get_min_sample_leaf_split(clf):
    '''
    input:
        clf: fitted estimator
    return:
        min_sample_leaf, min_sample_split in sklearn
        namely: min samples in leaves, min samples in nodes
    '''
    leaves_nodes = clf.tree_.children_left == -1
    father_nodes = clf.tree_.children_left > -1
    try:
        s = clf.tree_.n_node_samples
    except:
        s = clf.tree_.n_samples
    return s[leaves_nodes].min(), s[father_nodes].min()


############################# EXPORT ############################################

import warnings

from sklearn.externals import six

from sklearn.tree import _criterion




def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class Sentinel(object):
    def __repr__():
        return '"tree.dot"'
SENTINEL = Sentinel()


def export_graphviz(decision_tree, out_file=SENTINEL, max_depth=None,
                    feature_names=None, class_names=None, label='all',
                    filled=False, leaves_parallel=False, 
                    node_ids=False, proportion=False, rotate=False,
                    rounded=False, special_characters=False):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default='tree.dot')
        Handle or name of the output file. If ``None``, the result is
        returned as a string. This will the default from version 0.20.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, optional (default='all')
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, optional (default=False)
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, optional (default=False)
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.

    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, optional (default=False)
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, optional (default=False)
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    Returns
    -------
    dot_data : string
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """

    def get_color(value):
        # Find the appropriate color & intensity for a node
        if colors['bounds'] is None:
            # Classification tree
            color = list(colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = int(np.round(255 * (sorted_values[0] - sorted_values[1]) /
                                           (1 - sorted_values[1]), 0))
        else:
            # Regression tree or multi-output
            color = list(colors['rgb'][0])
            alpha = int(np.round(255 * ((value - colors['bounds'][0]) /
                                        (colors['bounds'][1] -
                                         colors['bounds'][0])), 0))

        # Return html color code in #RRGGBBAA format
        color.append(alpha)
        hex_codes = [str(i) for i in range(10)]
        hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
        color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

        return '#' + ''.join(color)

    def node_to_str(tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (label == 'root' and node_id == 0) or label == 'all'

        # PostScript compatibility for special characters
        if special_characters:
            characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>']
            node_string = '<'
        else:
            characters = ['#', '[', ']', '<=', '\\n', '"']
            node_string = '"'

        # Write node ID
        if node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           characters[3],
                                           round(tree.threshold[node_id], 4),
                                           characters[4])


        # Write node class distribution / regression value
        if proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, 4)
        elif proportion:
            # Classification
            value_text = np.around(value, 2)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, 4)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (class_names is not None and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if class_names is not True:
                class_name = class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + characters[5]

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if max_depth is None or depth <= max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                ranks['leaves'].append(str(node_id))
            elif str(depth) not in ranks:
                ranks[str(depth)] = [str(node_id)]
            else:
                ranks[str(depth)].append(str(node_id))

            out_file.write('%d [label=%s'
                           % (node_id,
                              node_to_str(tree, node_id, criterion)))

            if filled:
                # Fetch appropriate color for node
                if 'rgb' not in colors:
                    # Initialize colors and bounds if required
                    colors['rgb'] = _color_brew(tree.n_classes[0])
                    if tree.n_outputs != 1:
                        # Find max and min impurities for multi-output
                        colors['bounds'] = (np.min(-tree.impurity),
                                            np.max(-tree.impurity))
                    elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                        # Find max and min values in leaf nodes for regression
                        colors['bounds'] = (np.min(tree.value),
                                            np.max(tree.value))
                else:
                    # If multi-output color node by impurity
                    node_val = -tree.impurity[node_id]
                out_file.write(', fillcolor="%s"' % get_color(node_val))
            out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((rotate - .5) * -2)
                    out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        out_file.write('%d, headlabel="False"]' % angles[1])
                out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)

        else:
            ranks['leaves'].append(str(node_id))

            out_file.write('%d [label="(...)"' % node_id)
            if filled:
                # color cropped nodes grey
                out_file.write(', fillcolor="#C0C0C0"')
            out_file.write('] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    return_string = False
    try:
        if out_file == SENTINEL:
            warnings.warn("out_file can be set to None starting from 0.18. "
                          "This will be the default in 0.20.",
                          DeprecationWarning)
            out_file = "tree.dot"

        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, "w", encoding="utf-8")
            else:
                out_file = open(out_file, "wb")
            own_file = True

        if out_file is None:
            return_string = True
            out_file = six.StringIO()

        # The depth of each node for plotting with 'leaf' option
        ranks = {'leaves': []}
        # The colors to render each node with
        colors = {'bounds': None}

        out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = []
        if filled:
            rounded_filled.append('filled')
        if rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ", ".join(rounded_filled))
        if rounded:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if leaves_parallel:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if rounded:
            out_file.write('edge [fontname=helvetica] ;\n')
        if rotate:
            out_file.write('rankdir=LR ;\n')

        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        # If required, draw leaf nodes at same depth as each other
        if leaves_parallel:
            for rank in sorted(ranks):
                out_file.write("{rank=same ; " +
                               "; ".join(r for r in ranks[rank]) + "} ;\n")
        out_file.write("}")

        if return_string:
            return out_file.getvalue()

    finally:
        if own_file:
            out_file.close()




