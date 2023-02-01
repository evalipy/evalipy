import numpy as np


class Metrics:
    EPSILON = 1e-10

    def __init__(self) -> None:
        pass

    @staticmethod
    def error(y_true: np.ndarray, y_pred: np.ndarray):
        return y_true - y_pred

    @staticmethod
    def MSE(y_true, y_pred):
        return np.mean(np.square(Metrics.error(y_true, y_pred)))

    @staticmethod
    def MAE(y_true, y_pred):
        return np.mean(np.abs(Metrics.error(y_true, y_pred)))

    @staticmethod
    def R2(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + Metrics.EPSILON)

    @staticmethod
    def RMSE(y_true, y_pred):
        return np.sqrt(Metrics.MSE(y_true, y_pred))

    @staticmethod
    def RRSE(y_true, y_pred):
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

    @staticmethod
    def RAE(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + Metrics.EPSILON)
