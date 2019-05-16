__author__ = 'Mario Amrehn'

import sys
from copy import deepcopy
from enum import Enum, unique
from functools import partial
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
from zenlog import log
from sklearn import metrics as sklearn_metrics
from medpy.metric.binary import assd as medpy_assd, hd as medpy_hd  # , dc as medpy_dc
from medpy.metric.binary import precision as medpy_precision, recall as medpy_recall
from medpy.metric.binary import ravd as medpy_ravd
from medpy.metric.binary import obj_tpr as medpy_obj_tpr, obj_fpr as medpy_obj_fpr


@unique
class MetricEnums(Enum):
    ACC         = 'acc'           # accuracy classification score
    KAP         = 'kap'           # Cohen’s kappa, a statistic that measures inter-annotator agreement, [-1, 1]
    F1          = 'f1'            # balanced F-score or F-measure
    JAC         = 'jac'           # Jaccard similarity coefficient score
    LOG         = 'log'           # log loss, aka logistic loss or cross-entropy loss
    ASSD        = 'assd'          # average symmetric surface distance
    HD          = 'hd'            # hausdorff distance
    ARI         = 'ari'           # adjusted_rand_score
    MI          = 'mi'            # normalized_mutual_info_score
    HOM         = 'homogeneity'   # homogeneity_score
    COMPL       = 'completeness'  # completeness_score
    MSE         = 'mse'           # mean squared error
    V_MEASURE   = 'v_measure'     # v_measure
    ROC_AUC     = 'roc_auc'       # area under the receiver operating characteristic curve
    DICE        = 'dice'          # Sorensen-Dice coefficient
    PRECISION   = 'precison'      # precision; the inverse of the precision is recall.
    RECALL      = 'recall'        # recall; the inverse of the recall is precision.
    RAVD        = 'ravd'          # relative absolute volume difference
    OBJ_TPR     = 'obj_tpr'       # true positive rate of distinct binary object detection
    OBJ_FPR     = 'obj_fpr'       # false positive rate of distinct binary object detection
    FPR         = 'fpr'           # false positive rate
    FNR         = 'fnr'           # false negative rate


class Metrics:

    @staticmethod
    def get_default_parameters() -> Dict[str, Union[None, List[str]]]:
        return {
            'metrics': None  # None computes all metrics
        }

    @staticmethod
    def new_ordered_dict():
        return OrderedDict() if sys.version_info[:2] < (3, 6) else dict()

    def __init__(self):
        self.default_options = self.get_default_parameters()
        self.input_images = None

    def set_multiple_inputs(self, input_images):
        if 2 != len(input_images):
            raise ValueError('Error, metrics class expects two input elements')
        else:
            self.input_images = list(input_images)

    def get_outcome(self, options: Dict[str, Union[None, List[str]]] = {}) -> Dict[str, Union[float, type(np.nan)]]:
        opt = deepcopy(self.default_options)
        opt.update(options)

        y_true = self.input_images[0]  # gt
        y_pred = self.input_images[1]  # seg

        if str(y_true.dtype) != 'bool':
            y_true = 0 < y_true
        if str(y_pred.dtype) != 'bool':
            y_pred = 0 < y_pred

        metrics_ = self.new_ordered_dict()

        own_metrics = self._compute_metrics_own_implementation(y_true, y_pred, opt)
        metrics_.update({str(k): met for k, met in own_metrics.items()})

        sklearn_metrics = self._compute_metrics_sklearn(y_true, y_pred, opt)
        metrics_.update({str(k): -1 if (np.isnan(met) or met is None) else met
                         for k, met in sklearn_metrics.items()})

        medpy_metrics = self._compute_metrics_medpy(y_true, y_pred, opt)
        metrics_.update({str(k): -1 if (np.isnan(met) or met is None) else met
                         for k, met in medpy_metrics.items()})

        return metrics_

    @staticmethod
    def _compute_metrics_own_implementation(img0: np.ndarray, img1: np.ndarray,
                                            opt: Dict[str, Union[None, List[str]]]) -> Dict[Any, float]:

        def dice_coefficient(y_true, y_pred):
            if np.bool != y_true.dtype:
                y_true = 0 < y_true
            if np.bool != y_pred.dtype:
                y_pred = 0 < y_pred

            try:
                # medpy.metric.binary.dc takes about 3 times longer to compute than:
                return 2.0 * (np.count_nonzero(y_true & y_pred) / (np.count_nonzero(y_true) + np.count_nonzero(y_pred)))
            except ZeroDivisionError:
                return 1.0 if (np.count_nonzero(y_true) == int(0) == np.count_nonzero(y_pred)) else np.nan

        def fpr(y_true, y_pred):
            if np.bool != y_true.dtype:
                y_true = 0 < y_true
            if np.bool != y_pred.dtype:
                y_pred = 0 < y_pred
            return np.count_nonzero(y_true < y_pred) / y_true.size

        def fnr(y_true, y_pred):
            if np.bool != y_true.dtype:
                y_true = 0 < y_true
            if np.bool != y_pred.dtype:
                y_pred = 0 < y_pred
            return np.count_nonzero(y_true > y_pred) / y_true.size

        metrics = Metrics.new_ordered_dict()
        metrics[MetricEnums.DICE] = dice_coefficient  # Sorensen-Dice coefficient, best: 1.0, worst: 0.0
        metrics[MetricEnums.FPR] = fpr
        metrics[MetricEnums.FNR] = fnr

        metrics_to_evaluate = opt['metrics'] if opt['metrics'] is not None else metrics.keys()

        res = {k: met(img0, img1) for k, met in metrics.items() if k in metrics_to_evaluate}
        return res

    @staticmethod
    def _compute_metrics_sklearn(y_true: np.ndarray, y_pred: np.ndarray,
                                 opt: Dict[str, Union[None, List[str]]]) -> Dict[Any, float]:

        def roc_auc_score_(*argv, **kwargs):
            try:
                m = sklearn_metrics.roc_auc_score(*argv, **kwargs)
            except ValueError:  # Only one class present in y_true. ROC AUC score is not defined in that case.
                return np.nan
            return m

        def log_loss_(*argv, **kwargs):
            try:
                m = sklearn_metrics.log_loss(*argv, **kwargs)
            except ValueError:
                # y_true contains only one label (0).
                # Please provide the true labels explicitly through the labels argument.
                return np.nan
            return m

        metrics_to_evaluate = opt['metrics'] if opt['metrics'] is not None else \
            (MetricEnums.ACC, MetricEnums.KAP, MetricEnums.F1, MetricEnums.JAC, MetricEnums.LOG,
             MetricEnums.ARI, MetricEnums.MI, MetricEnums.HOM, MetricEnums.COMPL, MetricEnums.MSE,
             MetricEnums.ROC_AUC, MetricEnums.V_MEASURE)

        metrics = Metrics.new_ordered_dict()
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        metrics[MetricEnums.ACC] = sklearn_metrics.accuracy_score

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
        metrics[MetricEnums.KAP] = sklearn_metrics.cohen_kappa_score

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        metrics[MetricEnums.F1] = sklearn_metrics.f1_score

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score
        metrics[MetricEnums.JAC] = sklearn_metrics.jaccard_similarity_score

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
        metrics[MetricEnums.LOG] = log_loss_

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score
        metrics[MetricEnums.ARI] = sklearn_metrics.adjusted_rand_score  # adjusted rand index, best: 1.0, worst: 0.0

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
        # Normalized mutual information, best: 1.0, worst: <0, 0 == independent
        metrics[MetricEnums.MI] = partial(sklearn_metrics.normalized_mutual_info_score, average_method='arithmetic')

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
        metrics[MetricEnums.MSE] = sklearn_metrics.mean_squared_error  # mean squared error, best: 1.0, worst: 0.0

        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        metrics[MetricEnums.ROC_AUC] = roc_auc_score_  # Area under the receiver operating characteristic curve

        y_true, y_pred = y_true.ravel(), y_pred.ravel()

        if str(y_true.dtype) == 'bool':
            y_true = y_true.view(np.int8)
        else:
            y_true = (y_true > 0).view(np.int8)

        if str(y_pred.dtype) == 'bool':
            y_pred = y_pred.view(np.int8)
        else:
            y_pred = (y_pred > 0).view(np.int8)

        combined_measures = {MetricEnums.HOM, MetricEnums.COMPL, MetricEnums.V_MEASURE}
        if all((m in metrics_to_evaluate) for m in combined_measures):
            hom, compl, v_measure = sklearn_metrics.homogeneity_completeness_v_measure(y_true, y_pred)

            res = Metrics.new_ordered_dict()
            res[MetricEnums.HOM] = hom
            res[MetricEnums.COMPL] = compl
            res[MetricEnums.V_MEASURE] = v_measure

            metrics_to_evaluate = [m for m in metrics_to_evaluate if m not in combined_measures]
        else:
            res = Metrics.new_ordered_dict()
            additional_metrics = Metrics.new_ordered_dict()
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
            additional_metrics[MetricEnums.HOM] = sklearn_metrics.homogeneity_score  # homogeneity, best: 1.0, worst: <=0

            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
            additional_metrics[MetricEnums.COMPL] = sklearn_metrics.completeness_score  # completeness, best: 1.0, worst: <=0

            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
            additional_metrics[MetricEnums.V_MEASURE] = sklearn_metrics.v_measure_score  # v_measure, best: 1.0, worst: <=0

            for k in additional_metrics:
                if k in metrics_to_evaluate:
                    metrics[k] = additional_metrics[k]

        res.update({k: met(y_true, y_pred) for k, met in metrics.items() if k in metrics_to_evaluate})
        return res

    @staticmethod
    def _compute_metrics_medpy(y_true: np.ndarray, y_pred: np.ndarray,
                               opt: Dict[str, Union[None, List[str]]]) -> Dict[Any, float]:

        metrics_to_evaluate = opt['metrics'] if opt['metrics'] is not None else \
            (MetricEnums.ASSD, MetricEnums.HD, MetricEnums.PRECISION, MetricEnums.RECALL,
             MetricEnums.RAVD, MetricEnums.OBJ_TPR, MetricEnums.OBJ_FPR)  # MetricEnums.DICE,

        metrics_ = Metrics.new_ordered_dict()
        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.assd.html#medpy.metric.binary.assd
        metrics_[MetricEnums.ASSD] = lambda y_true_, y_pred_: medpy_assd(y_pred_.view(np.int8), y_true_.view(np.int8))

        # # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.hd.html#medpy.metric.binary.hd
        metrics_[MetricEnums.HD] = lambda y_true_, y_pred_: medpy_hd(y_pred_.view(np.int8), y_true_.view(np.int8))

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.dc.html#medpy.metric.binary.dc
        # metrics_[MetricEnums.DICE] = lambda y_true_, y_pred_: medpy_dc(y_pred_, y_true_),

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.precision.html#medpy.metric.binary.precision
        metrics_[MetricEnums.PRECISION] = lambda y_true_, y_pred_: medpy_precision(y_pred_, y_true_)

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.recall.html#medpy.metric.binary.recall
        metrics_[MetricEnums.RECALL] = lambda y_true_, y_pred_: medpy_recall(y_pred_, y_true_)

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.ravd.html#medpy.metric.binary.ravd
        metrics_[MetricEnums.RAVD] = lambda y_true_, y_pred_: medpy_ravd(y_pred_, y_true_)

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.obj_tpr.html#medpy.metric.binary.obj_tpr
        metrics_[MetricEnums.OBJ_TPR] = lambda y_true_, y_pred_: medpy_obj_tpr(y_pred_, y_true_)

        # http://pythonhosted.org/MedPy/generated/medpy.metric.binary.obj_fpr.html#medpy.metric.binary.obj_fpr
        metrics_[MetricEnums.OBJ_FPR] = lambda y_true_, y_pred_: medpy_obj_fpr(y_pred_, y_true_)

        if str(y_true.dtype) != 'bool' and -1 in y_true:
            y_true = y_true > 0

        if str(y_pred.dtype) != 'bool' and -1 in y_pred:
            y_pred = y_pred > 0

        try:
            res = {k: met(y_true, y_pred) for k, met in metrics_.items() if k in metrics_to_evaluate}
        except (RuntimeError, ZeroDivisionError):
            # RuntimeError: The second supplied array does not contain any binary object.
            # ZeroDivisionError: float division by zero
            # tmp = np.count_nonzero(img1)
            # if tmp is 0 or img1.size == tmp:
            res = Metrics.new_ordered_dict()
            for k, met in metrics_.items():
                if k in metrics_to_evaluate:
                    try:
                        res[k] = met(y_true, y_pred)
                    except Exception as ex:
                        log.warning(f'Metric {k} could not be computed: {ex}')
                        log.warning(f'y_true: {y_true.shape}, {y_true.dtype}, {np.unique(y_true)}')
                        log.warning(f'y_pred: {y_pred.shape}, {y_pred.dtype}, {np.unique(y_pred)}')
                        continue
        return res
