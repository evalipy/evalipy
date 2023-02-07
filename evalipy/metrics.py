import numpy as np
from math import sqrt


class RegressionMetrics:
    """
    Regression metrics class provides different metrics for assessing the regression model.
    Functions in this class are static and can be used independently.
    """

    EPSILON = 1e-10

    def __init__(self) -> None:
        pass

    @staticmethod
    def error(y_true: np.ndarray, y_pred: np.ndarray):
        return y_true - y_pred

    @staticmethod
    def MSE(y_true, y_pred):
        return np.mean(np.square(RegressionMetrics.error(y_true, y_pred)))

    @staticmethod
    def MAE(y_true, y_pred):
        return np.mean(np.abs(RegressionMetrics.error(y_true, y_pred)))

    @staticmethod
    def R2(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + RegressionMetrics.EPSILON)

    @staticmethod
    def RMSE(y_true, y_pred):
        return np.sqrt(RegressionMetrics.MSE(y_true, y_pred))

    @staticmethod
    def RRSE(y_true, y_pred):
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

    @staticmethod
    def RAE(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + RegressionMetrics.EPSILON)

    @staticmethod
    def NRMSE(y_true, y_pred):
        return RegressionMetrics.RMSE(y_true, y_pred) / (y_true.max() - y_true.min())

    @staticmethod
    def ME(y_true, y_pred):
        return np.mean(RegressionMetrics.error(y_true, y_pred))

    @staticmethod
    def MDAE(y_true, y_pred):
        return np.median(np.abs(RegressionMetrics.error(y_true, y_pred)))

    ALL_REGRESSION_METRICS = {"Mean Squared Error(MSE)": MSE,
                              "Mean Absolute Error(MAE)": MAE,
                              "R-Squared(R2)": R2,
                              "Root Mean Square Error(RMSE)": RMSE,
                              "Root Relative Squared Error(RRSE)": RRSE,
                              "Relative Absolute Error(RAE)": RAE,
                              "Normalized Root Mean Squared Error": NRMSE,
                              "Mean Error": ME,
                              "Median Absolute Error": MDAE,

                              }


class ClassificationMetrics:
    """
    Classification metrics class provides different metrics for assessing the classification model.
    Functions in this class are static and can be used independently.
    """

    def __init__(self):
        pass

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred, positive=1, negative=0):
        tp = np.sum(np.logical_and(y_pred == positive, y_true == positive))
        tn = np.sum(np.logical_and(y_pred == negative, y_true == negative))
        fp = np.sum(np.logical_and(y_pred == positive, y_true == negative))
        fn = np.sum(np.logical_and(y_pred == negative, y_true == positive))
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @staticmethod
    def precision(y_true, y_pred, positive=1, negative=0):
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return cm['tp'] / (cm['tp'] + cm['fp'])

    @staticmethod
    def reacll(y_true, y_pred, positive=1, negative=0):
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return cm['tp'] / (cm['tp'] + cm['fn'])

    @staticmethod
    def specificity(y_true, y_pred, positive=1, negative=0):
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return cm['tn'] / (cm['tn'] + cm['fp'])

    @staticmethod
    def npv(y_true, y_pred, positive=1, negative=0):
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return cm['tn'] / (cm['tn'] + cm['fn'])

    @staticmethod
    def accuracy_wcf(y_true, y_pred, positive=1, negative=0):
        # accuracy with confusion matrix
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return (cm['tp'] + cm['tn']) / (cm['fp'] + cm['tp'] + cm['tn'] + cm['fn'])

    @staticmethod
    def f1_score(y_true, y_pred, positive=1, negative=0):
        return 2 * (ClassificationMetrics.precision(y_true, y_pred, positive, negative) * (
            ClassificationMetrics.reacll(y_true, y_pred, positive, negative)) / (
                            ClassificationMetrics.precision(y_true, y_pred, positive, negative) + (
                        ClassificationMetrics.reacll(y_true, y_pred, positive, negative))))

    @staticmethod
    def balanced_accuracy(y_true, y_pred, positive=1, negative=0):
        return (ClassificationMetrics.reacll(y_true, y_pred, positive, negative) + ClassificationMetrics.specificity(
            y_true, y_pred, positive, negative)) / 2

    @staticmethod
    def MCC(y_true, y_pred, positive=1, negative=0):
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, positive, negative)
        return (cm['tp'] * cm['tn'] - cm['fp'] * cm['fn']) / sqrt(
            (cm['tp'] + cm['fp']) * (cm['tp'] + cm['fn']) * (cm['tn'] + cm['fp']) * (cm['tn'] + cm['fn']))

    ALL_CLASSIFICATION_METRICS = {'accuracy': accuracy}
