import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .metrics import Metrics
from .model import Model


class Report:
    def __init__(self, model: Model, actual_data: np.ndarray, predicted_data: np.ndarray, model_identifier='model 1'):
        self.model = model
        self.actual_data = actual_data
        self.predicted_data = predicted_data
        self.model_identifier = model_identifier
        self.report = self.__generate_report()

    def __generate_report(self):
        return pd.DataFrame.from_dict(
            {
                x: Metrics.ALL_METRICS[x](self.actual_data, self.predicted_data) for x in Metrics.ALL_METRICS.keys()

            }, orient='index', columns=[f'{self.model_identifier}'])

    def __str__(self) -> str:
        return self.report.__repr__()

    def __repr__(self) -> str:
        return self.report.__repr__()
