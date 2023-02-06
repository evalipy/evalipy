import numpy as np


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

    ALL_CLASSIFICATION_METRICS = {'accuracy': accuracy}
