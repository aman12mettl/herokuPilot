from sklearn import metrics as skmetrics
import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif
import xgboost as xgb # when calling the low level api


"""
Regression:=> use .predict()
Classification:=> use.predict() except auc/log_loss
auc/log_loss:= 
binary problem: (n_samples,)   .predict_proba()[:,1]
multiclass problem: (n_samples, n_classes)  .predict_proba()
true: [0,2,1,4,2] 1D array 
pred: [
    [0.1, 0.7, 0.2],
    [0.2, 0.3, 0.5],
    ...
]
    """

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    
def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False

def getaroom_metrics(y_true, y_pred):
    return max( 0, 100*(skmetrics.r2_score(y_true , y_pred)))

def amzcomp1_metrics(y_true, y_pred):
    return max( 0, 100*(skmetrics.r2_score(y_true , y_pred)))

## Keras Model Metrics
# https://stackoverflow.com/questions/57910680/how-to-name-custom-metrics-in-keras-fit-output
def my_metric_factory(the_param=1.0):
    def pauc(y_true, y_pred):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        if np.sum(y_true) == 0 or np.sum(y_true) == y_true.shape[0]:
            return 0.5
        # if np.sum(y_true) == 0:
        #     # all 0 : add one 1
        #     y_true = np.vstack([y_true, [1]]) #np.append(y_true, [[1]], axis=1)
        #     y_pred = np.vstack([y_pred, [1]])
        # elif np.sum(y_true) == y_true.shape[0]:
        #     # all 1 : add one 0
        #     y_true = np.vstack([y_true, [0]])
        #     y_pred = np.vstack([y_pred, [0]])
        return skmetrics.roc_auc_score(y_true, y_pred, max_fpr=0.05)
        # squared_difference = tf.square(y_true - y_pred)
        # return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

    pauc.__name__ = 'pauc'
    return pauc

# Custom Fbeta metrics with_eager = False 
# https://towardsdatascience.com/f-beta-score-in-keras-part-i-86ad190a252f
def binary_fbeta(ytrue , ypred, beta=1, threshold=0.5, epsilon=1e-7):
    # epsilon is set so as to avoid division by zero error
    
    beta_squared = beta**2 # squaring beta

    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(ypred, tf.constant(threshold)), tf.float32)

    tp = tf.reduce_sum(ytrue*ypred) # calculating true positives
    predicted_positive = tf.reduce_sum(ypred) # calculating predicted positives
    actual_positive = tf.reduce_sum(ytrue) # calculating actual positives
    
    precision = tp/(predicted_positive+epsilon) # calculating precision
    recall = tp/(actual_positive+epsilon) # calculating recall
    
    # calculating fbeta
    fb = (1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon)

    return fb

class StatefullBinaryFBeta(Metric):
  def __init__(self, name='fbeta1', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
    # initializing an object of the super class
    super(StatefullBinaryFBeta, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
    self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
    self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives

    # initializing other atrributes that wouldn't be changed for every object of this class
    self.beta_squared = beta**2 
    self.threshold = threshold
    self.epsilon = epsilon

  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
        
    self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
    self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
    self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute

  def result(self):
    self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
    self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall

    # calculating fbeta
    self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
    
    return self.fb

  def reset_states(self):
    self.tp.assign(0) # resets true positives to zero
    self.predicted_positive.assign(0) # resets predicted positives to zero
    self.actual_positive.assign(0) # resets actual positives to zero

class StatefullBinaryAccuracy(Metric):
  def __init__(self, name='acc1', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
    # initializing an object of the super class
    super(StatefullBinaryAccuracy, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
    self.tn = self.add_weight(name='tn', initializer='zeros')
    self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
    self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives

    # initializing other atrributes that wouldn't be changed for every object of this class
    self.beta_squared = beta**2 
    self.threshold = threshold
    self.epsilon = epsilon

  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
        
    self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
    self.tn.assign_add(tf.reduce_sum((1-ytrue)*(1-ypred)))
    self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
    self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute

  def result(self):
    self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
    self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall

    # calculating fbeta
    #self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
    
    self.fb = (self.tp + self.tn) / (self.tn + self.predicted_positive + self.actual_positive - self.tp + self.epsilon)
    return self.fb

  def reset_states(self):
    self.tp.assign(0) # resets true positives to zero
    self.tn.assign(0)
    self.predicted_positive.assign(0) # resets predicted positives to zero
    self.actual_positive.assign(0) # resets actual positives to zero



class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mae": self._mae,
            "mse": self._mse,
            "rmse": self._rmse,
            "msle": self._msle,
            "rmsle": self._rmsle,
            "r2": self._r2,
        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception(f"{metrics}: Metric not implemented")
        if metric == "mae":
            return self._mae(y_true=y_true, y_pred=y_pred)
        if metric == "mse":
            return self._mse(y_true=y_true, y_pred=y_pred)
        if metric == "rmse":
            return self._rmse(y_true=y_true, y_pred=y_pred)
        if metric == "msle":
            return self._msle(y_true=y_true, y_pred=y_pred)
        if metric == "rmsle":
            return self._rmsle(y_true=y_true, y_pred=y_pred)
        if metric == "r2":
            return self._r2(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true=y_true, y_pred=y_pred)


class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "fbeta": self._fbeta,
            "recall": self._recall,
            "precision": self._precision,
            "pauc": self._pauc,
            "auc": self._auc,
            "logloss": self._logloss,
            "auc_tf": self._auc_tf,
            "amex_metric": self.amex_metric,
        }

    # it allows to use an instance of this class as a function
    # a= Class..ics() then a("auc",y_true,y_pred)
    # y_pred is HARD CLASS 1,2,0,..
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        # For other metrics like precision, f1 we can pass y_proba and threh=0.5 is auto set 
        # but for below we have to pass proba and not o,1
        if metric not in self.metrics:
            raise Exception(f"{metric}: Metric not implemented")
        if metric == "pauc":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
        if metric == "auc":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc(y_true=y_true, y_pred=y_proba)
        if metric == "logloss":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc(y_true=y_true, y_pred=y_proba)
        if metric == "auc_tf":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self._auc_tf(y_true=y_true, y_pred=y_proba)
        if metric == "amex_metric":
            if y_proba is None:
                raise Exception(f"y_proba can't be None for {metric}")
            return self.amex_metric(y_true=y_true, y_pred=y_proba)
        else:
            # if it is a keras model than we will have probability here 
            # or just do > 0.5 it will automatically filter 
            y_proba = (y_proba>0.5).astype(int)
            return self.metrics[metric](y_true=y_true, y_pred=y_proba)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _fbeta(y_true, y_pred):
        return skmetrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5)

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _pauc(y_true, y_pred):
        # auc expects probability so we need y_proba
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred, max_fpr=0.05)
    
    @staticmethod
    def _auc(y_true, y_pred):
        # auc expects probability so we need y_proba
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc_tf(y_true, y_pred):
        # should have cuda enabled
        def fallback_auc(y_true, y_pred):
            try:
                return metrics.roc_auc_score(y_true, y_pred)
            except:
                return 0.5

        return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

    @staticmethod
    def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        def top_four_percent_captured(
            y_true: pd.DataFrame, y_pred: pd.DataFrame
        ) -> float:
            df = pd.concat([y_true, y_pred], axis="columns").sort_values(
                "prediction", ascending=False
            )
            df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
            four_pct_cutoff = int(0.04 * df["weight"].sum())
            df["weight_cumsum"] = df["weight"].cumsum()
            df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
            return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

        def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
            df = pd.concat([y_true, y_pred], axis="columns").sort_values(
                "prediction", ascending=False
            )
            df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
            df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
            total_pos = (df["target"] * df["weight"]).sum()
            df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
            df["lorentz"] = df["cum_pos_found"] / total_pos
            df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
            return df["gini"].sum()

        def normalized_weighted_gini(
            y_true: pd.DataFrame, y_pred: pd.DataFrame
        ) -> float:
            y_true_pred = y_true.rename(columns={"target": "prediction"})
            return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

        # sanity check
        y_true = pd.DataFrame(y_true, columns=["target"])
        y_pred = pd.DataFrame(y_pred, columns=["prediction"])
        #
        g = normalized_weighted_gini(y_true, y_pred)
        d = top_four_percent_captured(y_true, y_pred)

        return 0.5 * (g + d)

# @yunchonggan's fast metric implementation
# From https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

# ====================================================
# lgbmc amex metric
# ====================================================
# custom callback: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
def lgbmc_amex_metric(y_true, y_pred):
    # M1
    # Classification
    # cl = ClassificationMetrics()
    # return ("amex", cl("amex_metric",y_true,"y_pred_dummy",y_pred), True)

    # M2
    # https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)


# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

# ====================================================
# XGBOOST amex metric
# ====================================================
def xgboost_amex_metric_mod(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    return 'AMEXcustom', 1 - amex_metric_mod(y, predt)

def xgboost_amex_metric_mod1(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    return 'AMEXcustom', 1 - amzcomp1_metrics(y, predt)

# ====================================================
# Amex metric
# ====================================================
def amex_metric_lgb_base(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

# ====================================================
# LGB amex metric
# ====================================================
# https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7963
def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric_lgb_base(y_true, y_pred), True


# ====================================================
# amex custom metrics for keras
# ====================================================
from keras import backend as K
import tensorflow as tf

def amex_metric_tensorflow(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:

    # convert dtypes to float64
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    # count of positives and negatives
    n_pos = tf.math.reduce_sum(y_true)
    n_neg = tf.cast(tf.shape(y_true)[0], dtype=tf.float64) - n_pos

    # sorting by descring prediction values
    indices = tf.argsort(y_pred, axis=0, direction='DESCENDING')
    preds, target = tf.gather(y_pred, indices), tf.gather(y_true, indices)

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = tf.cumsum(weight / tf.reduce_sum(weight))
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = tf.reduce_sum(target[four_pct_filter]) / n_pos

    # weighted gini coefficient
    lorentz = tf.cumsum(target / n_pos)
    gini = tf.reduce_sum((lorentz - cum_norm_weight) * weight)

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


###########################################################
#                 cbc custom metrics
##############################################################
# https://stackoverflow.com/questions/65462220/how-to-create-custom-eval-metric-for-catboost
# https://www.kaggle.com/code/thedevastator/ensemble-lightgbm-catboost-xgboost
def amex_metric_cbc(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

class CustomMetric_cbc(object):
   def get_final_error(self, error, weight): return error
   def is_max_optimal(self): return True
   def evaluate(self, approxes, target, weight): return amex_metric_cbc(np.array(target), approxes[0]), 1.0