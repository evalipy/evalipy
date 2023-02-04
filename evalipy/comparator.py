import pandas as pd
from numpy import ndarray
from .model import Model
from .report import Report


class Comparator:
    def __init__(self, models: list[Model], x: ndarray, actual_data: ndarray) -> None:
        self.models = models
        self.actual_data = actual_data
        self.x = x
        self.result = self.__res()

    def __res(self):
        return pd.concat(
            [Report(i, self.actual_data, i.model.predict(self.x), i.model_type).report_DataFrame for i in self.models
             ],
            axis=1,
            ignore_index=False)

    def __str__(self):
        return self.result.__repr__()

    def __repr__(self):
        return self.result.__repr__()
