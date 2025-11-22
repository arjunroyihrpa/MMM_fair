# Author: Arjun Roy (arjun.roy@unibw.de, arjunroyihrpa@gmail.com) https://orcid.org/0000-0002-4279-9442 
# Apache License Version 2.0
import textwrap
import numpy as np
from time import time
from functools import partial
from numbers import Integral, Real
import itertools
from fairbench import v2 as fb
from tqdm import tqdm 
##sklearn ensemble
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting._gradient_boosting import _update_raw_predictions
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from sklearn._loss.loss import BaseLoss

#sklearn data processing and metrics
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _SCORERS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

#sklearn utils
from sklearn.utils import check_random_state, compute_sample_weight, resample
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state, compute_sample_weight, resample, compute_class_weight
from sklearn.utils.validation import (
    _check_monotonic_cst,
    _check_sample_weight,
    _check_y,
    _is_pandas_df,
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

from copy import deepcopy

from sklearn._loss import HalfBinomialLoss
from pymoo.mcdm.pseudo_weights import PseudoWeights
from typing import Literal

__all__ = ["MMM_Fair_GradientBoostedClassifier"]


def logistic_chain_rule(prob):
    """
    d(prob)/d(raw_pred) = prob * (1 - prob).
    """
    return prob * (1.0 - prob)


class MultiFairBinomialLoss(HalfBinomialLoss):
    """
    A multi-attribute fairness-aware binomial loss for binary classification.
    We combine multiple fairness penalties via a smooth max operator,
    approximating max across attributes in a differentiable manner.

    Each attribute is a binary array (0=protected,1=non-protected).
    We do e.g. 'DP' for each attribute => difference in predicted positives,
    then combine them via smoothmax with parameter 'beta'.

    Parameters
    ----------
    saIndex : array of protected values shape (n_samples,n_protected)
        For each protected attribute i, a value of 0/1 indicating group membership.
    alpha_fair : float, default=0.1
        Weight of the fairness penalty relative to logistic loss.
    beta : float, default=10.0
        Smoothmax parameter. Larger means closer to actual max but less smooth.
    fairness_type : str in {"DP","EP","EO","TPR","FPR"}, default="DP"
        Which measure to compute for each attribute.
        For multiple attributes, we do the same measure but you can adapt.
    sample_weight : array-like, optional
        For base logistic. 
    """

    def __init__(
        self,
        saIndex,
        saValue=None,
        alpha_fair=0.1,
        beta=10.0,
        fairness_type="DP",
        sample_weight=None,
    ):
        super().__init__(sample_weight=sample_weight)
        # store protected attributes as a list of arrays
        # each array shape (n_samples,)
        self.saIndex = saIndex
        if saValue is not None:
            self.saValue = saValue
        else:
            self.saValue = {}
            for j in range(self.saIndex.shape[-1]):
                self.saValue[str(j)]=0
        self.alpha_fair = alpha_fair
        self.beta = beta
        self.fairness_type = fairness_type

    def _compute_single_fairness(self, y_true, p, s):
        """
        Compute a single-attribute fairness measure:
        DP, EP, EO, TPR, FPR (toy version with squared difference).
        
        s is shape (n_samples,), 0=protected,1=non-protected
        p is predicted prob of pos class, shape (n_samples,).
        y_true is in {0,1}.
        Return:
          scalar fairness measure f_j
          array partial derivative df_j/d p[i] in shape (n_samples,).
        """
        idx0 = (s == 0)
        idx1 = (s == 1)
        n0 = max(1, idx0.sum())
        n1 = max(1, idx1.sum())

        if self.fairness_type.lower() == "dp":
            # difference in mean predicted positives
            mean0 = p[idx0].mean() if np.any(idx0) else 0.0
            mean1 = p[idx1].mean() if np.any(idx1) else 0.0
            diff = (mean1 - mean0)
            # penalty = diff^2
            penalty = diff*diff

            # derivative wrt each sample's p[i]
            grad = np.zeros_like(p)
            factor = 2.0 * diff
            for i in range(len(p)):
                if s[i] == 1:
                    chain = factor*(1./n1)
                else:
                    chain = factor*(-1./n0)
                grad[i] = chain * logistic_chain_rule(p[i])
            return penalty, grad

        elif self.fairness_type.lower() == "ep" or self.fairness_type.lower() == "tpr":
            # difference in TPR => subset y=1
            idx_pos = (y_true == 1)
            idx0p = idx0 & idx_pos
            idx1p = idx1 & idx_pos
            n0p = max(1, idx0p.sum())
            n1p = max(1, idx1p.sum())
            mean0p = p[idx0p].mean() if np.any(idx0p) else 0.
            mean1p = p[idx1p].mean() if np.any(idx1p) else 0.
            diff = mean1p - mean0p
            penalty = diff*diff

            grad = np.zeros_like(p)
            factor = 2.*diff
            for i in range(len(p)):
                if idx_pos[i]:
                    if s[i] == 1:
                        chain = factor*(1./n1p)
                    else:
                        chain = factor*(-1./n0p)
                    grad[i] = chain * logistic_chain_rule(p[i])
            return penalty, grad

        elif self.fairness_type.lower() == "fpr":
            # difference in FPR => subset y=0
            idx_neg = (y_true == 0)
            idx0n = idx0 & idx_neg
            idx1n = idx1 & idx_neg
            n0n = max(1, idx0n.sum())
            n1n = max(1, idx1n.sum())
            mean0n = (1.-p[idx0n]).mean() if np.any(idx0n) else 0.
            mean1n = (1.-p[idx1n]).mean() if np.any(idx1n) else 0.
            diff = mean1n - mean0n
            penalty = diff*diff

            grad = np.zeros_like(p)
            factor = 2.*diff
            for i in range(len(p)):
                if idx_neg[i]:
                    if s[i] == 1:
                        chain = factor*(1./n1n)*(-1.)
                    else:
                        chain = factor*(-1./n0n)*(-1.)
                    grad[i] = chain * logistic_chain_rule(p[i])
            return penalty, grad

        elif self.fairness_type.lower() == "eo":
            # TPR difference + TNR difference => for y=1, y=0
            # define penalty = (diff_tpr^2 + diff_tnr^2).
            idx_pos = (y_true == 1)
            idx_neg = (y_true == 0)
            idx1p = idx1 & idx_pos
            idx0p = idx0 & idx_pos
            idx1n = idx1 & idx_neg
            idx0n = idx0 & idx_neg

            n0p = max(1, idx0p.sum())
            n1p = max(1, idx1p.sum())
            n0n = max(1, idx0n.sum())
            n1n = max(1, idx1n.sum())

            mean0p = p[idx0p].mean() if np.any(idx0p) else 0.
            mean1p = p[idx1p].mean() if np.any(idx1p) else 0.
            diff_tpr = (mean1p - mean0p)

            # TNR => predicted negative => 1-p
            mean0n = (1.-p[idx0n]).mean() if np.any(idx0n) else 0.
            mean1n = (1.-p[idx1n]).mean() if np.any(idx1n) else 0.
            diff_tnr = (mean1n - mean0n)

            penalty = diff_tpr*diff_tpr + diff_tnr*diff_tnr

            grad = np.zeros_like(p)
            # partial derivative wrt TPR part
            factor_tpr = 2.*diff_tpr
            # partial derivative wrt TNR part
            factor_tnr = 2.*diff_tnr

            for i in range(len(p)):
                chain = 0.
                if y_true[i] == 1:
                    # TPR
                    if s[i] == 1:
                        chain += factor_tpr*(1./n1p)
                    else:
                        chain += factor_tpr*(-1./n0p)
                    # chain => derivative wrt p => multiply logistic_chain_rule
                    chain *= logistic_chain_rule(p[i])
                else:
                    # TNR => partial wrt p => negative sign
                    if s[i] == 1:
                        chain += factor_tnr*(1./n1n)*(-1.)
                    else:
                        chain += factor_tnr*(-1./n0n)*(-1.)
                    chain *= logistic_chain_rule(p[i])
                grad[i] = chain
            return penalty, grad

        else:
            # no fairness
            return 0., np.zeros_like(p)

    def _fairness_smoothmax(self, y_true, raw_pred):
        """
        For each attribute j:
          f_j, grad_j
        Then combine via softmax with param beta => smoothmax of f_j.

        Return total_penalty (scalar), grad_penalty (array shape=(n_samples,)) 
        """
        p = 1.0 / (1.0 + np.exp(-raw_pred))

        # For each attribute j => compute penalty_j, grad_j
        penalty_list = []
        grad_list = []
        for j in range(self.saIndex.shape[-1]):
            s_j=self.saIndex[:,j]
            pen_j, grad_j = self._compute_single_fairness(y_true, p, s_j)
            penalty_list.append(pen_j)
            grad_list.append(grad_j)
        
        # penalty_list => shape (m,) for m attributes
        # we define smoothmax( f_1, ..., f_m ) with param self.beta
        # => c = (1/beta)*log( sum( exp(beta * f_j) ) )
        # partial derivative wrt f_j => softmax weighting
        # partial wrt raw_pred => sum_j( w_j * d f_j/d raw_pred )
        f_arr = np.array(penalty_list, dtype=float)  # shape (m,)
        # avoid overflow
        max_f = np.max(f_arr)
        exps = np.exp(self.beta * (f_arr - max_f))
        sum_exps = np.sum(exps)
        c = (1.0/self.beta)*( np.log(sum_exps) ) + max_f  # shift back

        # derivative of c wrt f_j => w_j = exp( beta*(f_j - max_f) ) / sum_exps
        w = exps / sum_exps  # shape (m,)

        # partial derivative wrt each sample i => sum_j( w_j * grad_j[i] )
        # Combine all attributes' partial derivatives
        grad_c = np.zeros_like(grad_list[0])
        for j in range(self.saIndex.shape[-1]):
            grad_c += w[j] * grad_list[j]
        
        return c, grad_c

    def _fairness_penalty_grad(self, y_true, raw_pred):
        """
        The single-sample partial derivative of the multi-attribute
        fairness penalty. We do a smoothmax across attributes.

        Return grad_fair shape (n_samples,).
        """
        total_penalty, grad_penalty = self._fairness_smoothmax(y_true, raw_pred)
        return total_penalty, grad_penalty

    def gradient(self, y_true, raw_prediction, sample_weight, gradient_out, n_threads=1):
        """
        Overriding gradient: logistic + alpha_fair * fairness derivative
        ignoring hessian. scikit-learn calls gradient() if self.constant_hessian = True.
        """
        # call base logistic gradient
        super().gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
        # compute multi-attribute fairness
        _, grad_fair = self._fairness_penalty_grad(y_true, raw_prediction)
        gradient_out += self.alpha_fair * grad_fair

        return gradient_out

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight,
        gradient_out,
        hessian_out,
        n_threads=1
    ):
        """
        Overriding gradient & hessian. We'll do logistic for hessian, and
        add alpha_fair * fairness to gradient.
        """

        super().gradient_hessian(
            y_true, raw_prediction, sample_weight, gradient_out, hessian_out, n_threads
        )

        _, grad_fair = self._fairness_penalty_grad(y_true, raw_prediction)

        if len(gradient_out.shape)<len(grad_fair.shape):
            grad_fair = grad_fair.ravel()

        mean_log = np.mean(np.abs(gradient_out))
        mean_fair = np.mean(np.abs(grad_fair))
        ratio = mean_log / (mean_fair + 1e-7) ##slack to avoid division by zero
        gradient_out += self.alpha_fair * grad_fair*ratio

        return gradient_out,hessian_out



class MMM_Fair_GradientBoostedClassifier(HistGradientBoostingClassifier):
    def __init__(
        self,
        saIndex=None,
        saValue=None,
        constraint='DP',
        alpha=0.1,
        store_iter_metrics=True,
        preference=None,
        # plus any standard HistGradientBoostingClassifier params
        early_stopping="auto",
        validation_fraction=0.1,
        max_iter=100,
        random_state=None,
        # ...
        **kwargs
    ):

        if constraint is None:
            # your default or fallback
            super().__init__(
                loss="log_loss",
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                max_iter=max_iter,
                random_state=random_state,
                **kwargs
            )
            self.saValue = saValue
            self.saIndex = saIndex
        else:
            if saIndex is None:
                raise ValueError("Protected features cannot be None for fairness intervention")
            else:
                self.saIndex = saIndex
                if saValue is not None:
                    self.saValue = saValue
                else:
                    self.saValue = {}
                    for j in range(self.saIndex.shape[-1]):
                        self.saValue[str(j)]=0
                self.sensitives=list(self.saValue.keys())
            super().__init__(
                loss=MultiFairBinomialLoss(
                            saIndex=self.saIndex,
                            saValue=self.saValue,
                            alpha_fair=alpha,
                            beta=10.0,
                            fairness_type=constraint,  # or "EP","EO", "TPR", "FPR"
                        ),
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                max_iter=max_iter,
                random_state=random_state,
                **kwargs
            )
        self.alpha=alpha
        self.constraint=constraint

        if self.saValue is None or len(self.saValue)!=self.saIndex.shape[-1]:
            if saIndex is not None:
                self.saValue = {}
                for j in range(self.saIndex.shape[-1]):
                        self.saValue[str(j)]=0
                self.sensitives=list(self.saValue.keys())
            else:
                self.sensitives=[]
        self.store_iter_metrics = store_iter_metrics
        self.acc_loss_ = []
        self.bal_loss_ = []
        self.mmm_loss_ = []
        self.all_estimators=[]
        self.ob=[]
        self.feat_obs=[]
        self.fairobs=[]
        self.theta=-1
        self.pseudo=None
        self.pareto=False
        self.PF=None
        self.preference=preference
        

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,) default=None
            Weights of training data.
    
            
    
        Returns
        -------
        self : object
            Fitted estimator.
        """
        fit_start_time = time()
        acc_find_split_time = 0.0  # time spent finding the best splits
        acc_apply_split_time = 0.0  # time spent splitting nodes
        acc_compute_hist_time = 0.0  # time spent computing histograms
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.0
        X, known_categories = self._preprocess_X(X, reset=True)
        y = _check_y(y, estimator=self)
        y = self._encode_y(y)
        check_consistent_length(X, y)
        if self.saIndex is None:
            self.saIndex = np.random.randint(2, size=(len(X),1))
            print(f"Caution!! saIndex not provided. Model initiated with random protected values, sample: {self.saIndex[:50,0]}")
            
        check_consistent_length(X, self.saIndex)
        # Do not create unit sample weights by default to later skip some
        # computation
        
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            # TODO: remove when PDP supports sample weights
            self._fitted_with_sw = True
    
        sample_weight = self._finalize_sample_weight(sample_weight, y)
    
        rng = check_random_state(self.random_state)
    
        # When warm starting, we want to reuse the same seed that was used
        # the first time fit was called (e.g. train/val split).
        # For feature subsampling, we want to continue with the rng we started with.
        if not self.warm_start or not self._is_fitted():
            self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
            feature_subsample_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
            self._feature_subsample_rng = np.random.default_rng(feature_subsample_seed)
    
        self._validate_parameters()
        monotonic_cst = _check_monotonic_cst(self, self.monotonic_cst)
        # _preprocess_X places the categorical features at the beginning,
        # change the order of monotonic_cst accordingly
        if self.is_categorical_ is not None:
            monotonic_cst_remapped = np.concatenate(
                (
                    monotonic_cst[self.is_categorical_],
                    monotonic_cst[~self.is_categorical_],
                )
            )
        else:
            monotonic_cst_remapped = monotonic_cst
    
        # used for validation in predict
        n_samples, self._n_features = X.shape
    
        # Encode constraints into a list of sets of features indices (integers).
        interaction_cst = self._check_interaction_cst(self._n_features)
        self._in_fit = True
    
        # `_openmp_effective_n_threads` is used to take cgroups CPU quotes
        # into account when determine the maximum number of threads to use.
        n_threads = _openmp_effective_n_threads()
    
    
        if isinstance(self.loss, str):
            self._loss = self._get_loss(sample_weight=sample_weight)
        elif isinstance(self.loss, BaseLoss):
            self._loss = self.loss
    
        if self.early_stopping == "auto":
            self.do_early_stopping_ = n_samples > 10000
        else:
            self.do_early_stopping_ = self.early_stopping
    
        # create validation data if needed
        self._use_validation_data = self.validation_fraction is not None
        if self.do_early_stopping_ and self._use_validation_data:
            # stratify for classification
            # instead of checking predict_proba, loss.n_classes >= 2 would also work
            stratify = y if hasattr(self._loss, "predict_proba") else None
    
            # Save the state of the RNG for the training and validation split.
            # This is needed in order to have the same split when using
            # warm starting.
    
            if sample_weight is None:
                X_train, X_val, y_train, y_val, sa_train, sa_val = train_test_split(
                    X,
                    y,
                    self.saIndex,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                    random_state=self._random_seed,
                )
                sample_weight_train = sample_weight_val = None
            else:
                # TODO: incorporate sample_weight in sampling here, as well as
                # stratify
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sa_train,
                    sa_val,
                    sample_weight_train,
                    sample_weight_val,
                ) = train_test_split(
                    X,
                    y,
                    self.saIndex,
                    sample_weight,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                    random_state=self._random_seed,
                )
        else:
            X_train, y_train, sa_train, sample_weight_train = X, y, self.saIndex, sample_weight
            X_val = y_val = sa_val = sample_weight_val = None
    
    
        n_bins = self.max_bins + 1  # + 1 for missing values
        self._bin_mapper = _BinMapper(
            n_bins=n_bins,
            is_categorical=self._is_categorical_remapped,
            known_categories=known_categories,
            random_state=self._random_seed,
            n_threads=n_threads,
        )
        X_binned_train = self._bin_data(X_train, is_training_data=True)
        if X_val is not None:
            X_binned_val = self._bin_data(X_val, is_training_data=False)
        else:
            X_binned_val = None
    
        # Uses binned data to check for missing values
        has_missing_values = (
            (X_binned_train == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )
    
        if self.verbose:
            print("Fitting gradient boosted rounds:")
    
        n_samples = X_binned_train.shape[0]
        scoring_is_predefined_string = self.scoring in _SCORERS
        need_raw_predictions_val = X_binned_val is not None and (
            scoring_is_predefined_string or self.scoring == "loss"
        )
        
        # First time calling fit, or no warm start
        if not (self._is_fitted() and self.warm_start):
            # Clear random state and score attributes
            self._clear_state()
    
            # initialize raw_predictions: those are the accumulated values
            # predicted by the trees for the training data. raw_predictions has
            # shape (n_samples, n_trees_per_iteration) where
            # n_trees_per_iterations is n_classes in multiclass classification,
            # else 1.
            # self._baseline_prediction has shape (1, n_trees_per_iteration)
            self._baseline_prediction = self._loss.fit_intercept_only(
                y_true=y_train, sample_weight=sample_weight_train
            ).reshape((1, -1))
            raw_predictions = np.zeros(
                shape=(n_samples, self.n_trees_per_iteration_),
                dtype=self._baseline_prediction.dtype,
                order="F",
            )
            raw_predictions += self._baseline_prediction
    
            # predictors is a matrix (list of lists) of TreePredictor objects
            # with shape (n_iter_, n_trees_per_iteration)
            self._predictors = predictors = []
    
            # Initialize structures and attributes related to early stopping
            self._scorer = None  # set if scoring != loss
            raw_predictions_val = None  # set if use val and scoring is a string
            self.train_score_ = []
            self.validation_score_ = []
            
            
        
            if self.do_early_stopping_:
                # populate train_score and validation_score with the
                # predictions of the initial model (before the first tree)
    
                # Create raw_predictions_val for storing the raw predictions of
                # the validation data.
                if need_raw_predictions_val:
                    raw_predictions_val = np.zeros(
                        shape=(X_binned_val.shape[0], self.n_trees_per_iteration_),
                        dtype=self._baseline_prediction.dtype,
                        order="F",
                    )
    
                    raw_predictions_val += self._baseline_prediction
    
    
                ##self.scoring can be accuracy, loss, etc.
                """
                from sklearn.metrics import make_scorer, accuracy_score, mean_squared_log_error
                scoring = {
                        "accuracy": make_scorer(accuracy_score),
                        "mean_squared_log_error": make_scorer(mean_squared_log_error),
                    }
    
                """
    
                if self.scoring == "loss":
                    # we're going to compute scoring w.r.t the loss. As losses
                    # take raw predictions as input (unlike the scorers), we
                    # can optimize a bit and avoid repeating computing the
                    # predictions of the previous trees. We'll reuse
                    # raw_predictions (as it's needed for training anyway) for
                    # evaluating the training loss.
    
                    self._check_early_stopping_loss(
                        raw_predictions=raw_predictions,
                        y_train=y_train,
                        sa_train=sa_train,
                        sample_weight_train=sample_weight_train,
                        raw_predictions_val=raw_predictions_val,
                        y_val=y_val,
                        sa_val=sa_val,
                        sample_weight_val=sample_weight_val,
                        n_threads=n_threads,
                    )
                else:
                    self._scorer = check_scoring(self, self.scoring)
                    # _scorer is a callable with signature (est, X, y) and
                    # calls est.predict() or est.predict_proba() depending on
                    # its nature.
                    # Unfortunately, each call to _scorer() will compute
                    # the predictions of all the trees. So we use a subset of
                    # the training set to compute train scores.
    
                    # Compute the subsample set
                    (
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                        indices_small_train,
                    ) = self._get_small_trainset(
                        X_binned_train,
                        y_train,
                        sample_weight_train,
                        self._random_seed,
                    )
    
                    # If the scorer is a predefined string, then we optimize
                    # the evaluation by re-using the incrementally updated raw
                    # predictions.
                    if scoring_is_predefined_string:
                        raw_predictions_small_train = raw_predictions[
                            indices_small_train
                        ]
                    else:
                        raw_predictions_small_train = None
    
                    self._check_early_stopping_scorer(
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                        X_binned_val,
                        y_val,
                        sample_weight_val,
                        raw_predictions_small_train=raw_predictions_small_train,
                        raw_predictions_val=raw_predictions_val,
                    )
            begin_at_stage = 0
    
    
          
    
        # warm start: this is not the first time fit was called
        else:
            # Check that the maximum number of iterations is not smaller
            # than the number of iterations from the previous fit
            if self.max_iter < self.n_iter_:
                raise ValueError(
                    "max_iter=%d must be larger than or equal to "
                    "n_iter_=%d when warm_start==True" % (self.max_iter, self.n_iter_)
                )
    
            # Convert array attributes to lists
            self.train_score_ = self.train_score_.tolist()
            self.validation_score_ = self.validation_score_.tolist()
    
            # Compute raw predictions
            raw_predictions = self._raw_predict(X_binned_train, n_threads=n_threads)
            if self.do_early_stopping_ and need_raw_predictions_val:
                raw_predictions_val = self._raw_predict(
                    X_binned_val, n_threads=n_threads
                )
            else:
                raw_predictions_val = None
    
            if self.do_early_stopping_ and self.scoring != "loss":
                # Compute the subsample set
                (
                    X_binned_small_train,
                    y_small_train,
                    sample_weight_small_train,
                    indices_small_train,
                ) = self._get_small_trainset(
                    X_binned_train, y_train, sample_weight_train, self._random_seed
                )
                sa_train=self.saIndex[indices_small_train]
    
            # Get the predictors from the previous fit
            predictors = self._predictors
    
            begin_at_stage = self.n_iter_
    
        # initialize gradients and hessians (empty arrays).
        # shape = (n_samples, n_trees_per_iteration).
        gradient, hessian = self._loss.init_gradient_and_hessian(
            n_samples=n_samples, dtype=G_H_DTYPE, order="F"
        )
    
    
        progress_bar = tqdm(range(begin_at_stage, self.max_iter), desc="MMM_Fair gradboost", unit="round")
        for iteration in progress_bar:#range(begin_at_stage, self.max_iter):
            if self.verbose >= 2:
                iteration_start_time = time()
                print(
                    "[{}/{}] ".format(iteration + 1, self.max_iter), end="", flush=True
                )
    
            # Update gradients and hessians, inplace
            # Note that self._loss expects shape (n_samples,) for
            # n_trees_per_iteration = 1 else shape (n_samples, n_trees_per_iteration).
            if self._loss.constant_hessian:
                self._loss.saIndex=sa_train
                self._loss.gradient(
                    y_true=y_train,
                    raw_prediction=raw_predictions,
                    sample_weight=sample_weight_train,
                    gradient_out=gradient,
                    n_threads=n_threads,
                )
            else:
                self._loss.saIndex=sa_train
                self._loss.gradient_hessian(
                    y_true=y_train,
                    raw_prediction=raw_predictions,
                    sample_weight=sample_weight_train,
                    gradient_out=gradient,
                    hessian_out=hessian,
                    n_threads=n_threads,
                )
    
            # Append a list since there may be more than 1 predictor per iter
            predictors.append([])
    
            # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
            # on gradient and hessian to simplify the loop over n_trees_per_iteration_.
            if gradient.ndim == 1:
                g_view = gradient.reshape((-1, 1))
                h_view = hessian.reshape((-1, 1))
            else:
                g_view = gradient
                h_view = hessian
    
            # Build `n_trees_per_iteration` trees.
            for k in range(self.n_trees_per_iteration_):
                grower = TreeGrower(
                    X_binned=X_binned_train,
                    gradients=g_view[:, k],
                    hessians=h_view[:, k],
                    n_bins=n_bins,
                    n_bins_non_missing=self._bin_mapper.n_bins_non_missing_,
                    has_missing_values=has_missing_values,
                    is_categorical=self._is_categorical_remapped,
                    monotonic_cst=monotonic_cst_remapped,
                    interaction_cst=interaction_cst,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    feature_fraction_per_split=self.max_features,
                    rng=self._feature_subsample_rng,
                    shrinkage=self.learning_rate,
                    n_threads=n_threads,
                )
                grower.grow()
    
                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time
                acc_compute_hist_time += grower.total_compute_hist_time
    
                ### _update_leaves_values only uses loss.fit_intercept_only, so saIndex not needed
                if not self._loss.differentiable:
                    _update_leaves_values(
                        loss=self._loss,
                        grower=grower,
                        y_true=y_train,
                        raw_prediction=raw_predictions[:, k],
                        sample_weight=sample_weight_train,
                    )
    
                predictor = grower.make_predictor(
                    binning_thresholds=self._bin_mapper.bin_thresholds_
                )
                predictors[-1].append(predictor)
    
                # Update raw_predictions with the predictions of the newly
                # created tree.
                tic_pred = time()
                _update_raw_predictions(raw_predictions[:, k], grower, n_threads)
                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred
    
    
    
            should_early_stop = False
            if self.do_early_stopping_:
                # Update raw_predictions_val with the newest tree(s)
                if need_raw_predictions_val:
                    for k, pred in enumerate(self._predictors[-1]):
                        raw_predictions_val[:, k] += pred.predict_binned(
                            X_binned_val,
                            self._bin_mapper.missing_values_bin_idx_,
                            n_threads,
                        )
    
                if self.scoring == "loss":
                    should_early_stop = self._check_early_stopping_loss(
                        raw_predictions=raw_predictions,
                        y_train=y_train,
                        sa_train=sa_train,
                        sample_weight_train=sample_weight_train,
                        raw_predictions_val=raw_predictions_val,
                        y_val=y_val,
                        sa_val=sa_val,
                        sample_weight_val=sample_weight_val,
                        n_threads=n_threads,
                    )
    
                else:
                    # If the scorer is a predefined string, then we optimize the
                    # evaluation by re-using the incrementally computed raw predictions.
                    if scoring_is_predefined_string:
                        raw_predictions_small_train = raw_predictions[
                            indices_small_train
                        ]
                    else:
                        raw_predictions_small_train = None
    
                    should_early_stop = self._check_early_stopping_scorer(
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                        X_binned_val,
                        y_val,
                        sample_weight_val,
                        raw_predictions_small_train=raw_predictions_small_train,
                        raw_predictions_val=raw_predictions_val,
                    )
                    ####hola
    
    
            if self.verbose >= 2:
                self._print_iteration_stats(iteration_start_time)
    
            # maybe we could also early stop if all the trees are stumps?
            if should_early_stop:
                break
    
        if self.verbose:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self._predictors
                for predictor in predictors_at_ith_iteration
            )
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self._predictors
            )
            print(
                "Fit {} trees in {:.3f} s, ({} total leaves)".format(
                    n_predictors, duration, n_total_leaves
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent computing histograms:", acc_compute_hist_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent finding best splits:", acc_find_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent applying splits:", acc_apply_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format("Time spent predicting:", acc_prediction_time)
            )
    
        self.train_score_ = np.asarray(self.train_score_)
        self.validation_score_ = np.asarray(self.validation_score_)
        self.all_estimators=[predictor for predictor in self._predictors]
        
    
        if self.store_iter_metrics:
            X_ = check_array(X, dtype=None, accept_sparse=False)
            check_is_fitted(self)
            for ypred in self.staged_predict(X_):
                o1=1-accuracy_score(y_pred=ypred,y_true=y)
                o2=1-balanced_accuracy_score(y_pred=ypred,y_true=y)#.item()
                o3=[]
                for i in range(self.saIndex.shape[-1]):
                    sens = fb.categories(self.saIndex[:, i])
                    report = fb.reports.pairwise(
                        predictions=ypred,
                        labels=y,
                        sensitive=sens
                    )
                    report_dict=report.show(env=fb.export.ToDict)
                    dp=report_dict['depends'][6]['depends'][1]['value']['value']
                    ep=report_dict['depends'][6]['depends'][2]['value']['value']
                    tpr = report_dict['depends'][6]['depends'][2]['value']['value']
                    fpr = report_dict['depends'][6]['depends'][3]['value']['value']
                    eo=max(report_dict['depends'][6]['depends'][2]['value']['value'],
                                      report_dict['depends'][6]['depends'][3]['value']['value'])
                    if self.constraint.lower()=='dp':
                        o3.append(dp)
                    elif self.constraint.lower()=='ep':
                        o3.append(ep)
                    elif self.constraint.lower()=='eo':
                        o3.append(eo)
                    elif self.constraint.lower()=='tpr':
                        o3.append(tpr)
                    elif self.constraint.lower()=='fpr':
                        o3.append(fpr)
                    else:
                        o3.append(0.99)
                self.ob.append([o1, o2, max(o3)])
                self.feat_obs.append(o3) 
                self.fairobs.append([dp,ep,eo,tpr,fpr])
                
                
        self.ob = np.array(self.ob)
        self.feat_obs = np.array(self.feat_obs)
        self.fairobs = np.array(self.fairobs)
        del self._in_fit  # hard delete so we're sure it can't be used anymore
        return self
    
    
    def update_theta(
        self,
        criteria: Literal["all", "fairness","fairdefs"] = "all",
        preference=[0.33, 0.34, 0.33],
        theta=None ##for bruteforce theta update
    ):
        def is_pareto(costs, maximise=False):
            """
            :param costs: An (n_points, n_costs) array
            :maximise: boolean. True for maximising, False for minimising
            :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
            """
            eps = 0.001
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    if maximise:
                        is_efficient[is_efficient] = np.any(
                            costs[is_efficient] >= c, axis=1
                        )  # Remove dominated points
                    else:
                        is_efficient[is_efficient] = np.any(
                            costs[is_efficient] <= c, axis=1
                        )  # Remove dominated points
            return is_efficient

        if theta is None:
            #self.preference = preference
            best_theta = 0
            if criteria.lower() == "fairness":
                objective = deepcopy(self.feat_obs)
                if len(preference)!= self.feat_obs.shape[-1]:
                    preference = [1/self.feat_obs.shape[-1] for i in range(self.feat_obs.shape[-1])]
            elif criteria.lower() == "fairdefs":
                objective = deepcopy(self.fairobs)
            else:
                objective = deepcopy(self.ob)
            # objective=np.round(objective,2)
            if self.pareto == False:
                PF = {i: objective[i] for i in range(len(objective)) if np.any(objective[i]==0)==False}
                F = np.array([objective[o] for o in range(len(objective))])
                self.PF = PF
            else:
                pf = is_pareto(objective)
                PF = {i: objective[i] for i in range(len(pf)) if pf[i] == True and np.any(objective[i]==0)==False}
                F = np.array(list(PF.values()))
                self.PF = PF
    
            if self.preference is None or len(self.preference)!= objective.shape[-1]:
                weights = preference  ##Preference Weights
            else:
                weights = self.preference
    
            best_theta, pseudo_weights = PseudoWeights(weights).do(
                F, return_pseudo_weights=True
            )
    
            if self.preference == None:
                sum_W = [sum((1 - pseudo_weights[w]) * F[w]) for w in range(len(PF))]
                best_theta = sum_W.index(min(sum_W))
    
            self.theta = list(PF.keys())[best_theta] + 1
            self.pseudo = pseudo_weights
            self._predictors=self.all_estimators[:self.theta]
        else:
            self.theta =theta
            self._predictors=self.all_estimators[:self.theta]
    
    def _check_early_stopping_loss(
        self,
        raw_predictions,
        y_train,
        sa_train,
        sample_weight_train,
        raw_predictions_val,
        y_val,
        sa_val,
        sample_weight_val,
        n_threads=1,
    ):
        """Check if fitting should be early-stopped based on loss.
    
        Scores are computed on validation data or on training data.
        """
        self._loss.saIndex=sa_train
        self.train_score_.append(
            -self._loss(
                y_true=y_train,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight_train,
                n_threads=n_threads,
            )
        )
    
        if self._use_validation_data:
            self._loss.saIndex=sa_val
            self.validation_score_.append(
                -self._loss(
                    y_true=y_val,
                    raw_prediction=raw_predictions_val,
                    sample_weight=sample_weight_val,
                    n_threads=n_threads,
                )
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)
    
    def see_pareto(self):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        def plot2d(x,y,axis_names):
            pass
        def plot3d(x = [1, 2, 3, 4, 5],y = [2, 3, 1, 4, 5],z = [3, 1, 2, 5, 4],theta=[1,2,3,4,5],criteria='all',axis_names=['X','Y','Z'], title="3D Scatter Plot of Pareto front.", desc=(
                    "🔹 This plot visualizes our data in 3D.<br>"
                    "🔹 Each point's position is determined by X, Y, Z coordinates.<br>"
                    "🔹 Marker color is blue by default, but could represent categories.<br>"
                    "🔹 Hover over points to see info about the associated **theta** (ensemble pointer).<br>"
                    "🔹 Theta can be used as preferences to update the model:<br>"
                    "➡ <b>Example: model.update_theta(theta=my_preferred_theta)</b>"
            "<br> _________________________________________________________ <br> "
                ) ):
            # Create the figure
            title_wrap=100
            wrapped_title = "<br>".join(textwrap.wrap(title, width=title_wrap))
            hidden_data=[f'Theta: {theta[i]}' for i in range(len(theta))]
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
            color_values = np.array(x) + np.array(y) + np.array(z)
            color_label = "Sum of losses (darker better)"
            # Add 3D scatter plot
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=8,
                                                color=color_values,  # Color based on the chosen attribute
                                                colorscale="Viridis",  # Try 'Plasma', 'Cividis', 'Inferno', etc.
                                                showscale=True,
                                                colorbar=dict(title=color_label),
                                            ), name='Points',text=hidden_data, customdata=hidden_data ), 
                          row=1, col=1)
            
            # Define callback function for selection event
            def update_point(trace, points, selector):
                selected_points = [(trace.x[i], trace.y[i], trace.z[i]) for i in points.point_inds]
                selected_hidden_data = [trace.customdata[i] for i in points.point_inds]
                print("Selected Points:", selected_points,selected_hidden_data)
            
            # Assign callback function to plot
            fig.data[0].on_selection(update_point)
            
            # Update layout
            fig.update_layout(title=dict(
                                    text=f"{criteria}-objectives<br>{wrapped_title}",
                                    #x=0.5,  # Center the title
                                    #y=1.2,  # Push title down slightly to prevent clipping
                                    font=dict(size=18),
                                ),
                              scene=dict(
                                        xaxis_title='X:'+axis_names[0],
                                        yaxis_title='Y:'+ axis_names[1],
                                        zaxis_title='Z:'+ axis_names[2]
                                    ),
                             annotations=[
                                            go.layout.Annotation(
                                            showarrow=False,
                                            text=desc,
                                            x=1.02,  # text to the right outside the plot
                                            y=1.02,
                                            xref="paper",
                                            yref="paper",
                                            font=dict(size=14, color="black"),
                                            bgcolor="rgba(255, 255, 255, 0.85)",  
                                            bordercolor="black",
                                            borderwidth=2,
                                            borderpad=10,
                                            align="left",
                                        )
                                                                    ],
                                    width=1300,  # width of the plot
                                    height=700 )
            
            # Show plot
            fig.show()
        
        #self.update_theta(criteria='all')
        PF=np.array([self.ob[i] for i in range(len(self.ob))])
        thetas=np.arange(len(self.ob))
        title=f"3D Scatter Plot. Showing various trade-off points between Accuracy, Balanced Accuracy, and Maximum violation of {self.constraint} fairness among protected attributes."
        plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria="Multi",
               axis_names=['Acc.','Balanc. Acc', 'MMM-fair'],title=title)
        PF=np.array([self.fairobs[i] for i in range(len(self.fairobs))])
        title=f"3D Scatter Plot. Showing various trade-off points between maximum violation of Demopgraphic Parity, Equal Opportunity, and Equalized odds fairness for the given set of protected attributes."
        plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas, criteria= "Multi-definitions",
               axis_names=['DP','EqOpp', 'EqOdd'],title=title)
        if len(self.sensitives)>0:
            #self.theta=None
            #self.update_theta(criteria='fairness')
            PF=np.array([self.feat_obs[i] for i in range(len(self.feat_obs))])
            title=f"3D Scatter Plot. Showing various trade-off points between violation of {self.constraint} fairness among the protected attributes {self.sensitives}."
            if PF.shape[-1]>2:
                plot3d(x=PF[:,0],y=PF[:,1],z=PF[:,2], theta=thetas,
                       criteria='Multi-attribute',axis_names=self.sensitives,title=title)
            elif PF.shape[-1]==2:
                    plot3d(x=PF[:,0],y=PF[:,1],z=np.zeros_like(PF[:,1]), theta=thetas,
                       criteria='Multi-attribute',axis_names=self.sensitives+[''],title=title)
        
            else:
                print('Not a MMM-fair')   
    
    


