import pandas as pd
from .report import Report


class Comparator:
    def __init__(self, model_A, model_B, actual_data, predicted_data_A, predicted_data_B) -> None:
        self.model_A = model_A
        self.model_B = model_B
        self.actual_data = actual_data
        self.predicted_data_A = predicted_data_A
        self.predicted_data_B = predicted_data_B
        self.result = self.__res()

    def __res(self):
        return pd.concat(
            [Report(self.model_A, self.actual_data, self.predicted_data_A, model_identifier='model A').report,
             Report(self.model_B, self.actual_data, self.predicted_data_B, model_identifier='model B').report],
            axis=1,
            ignore_index=False)

    def __str__(self):
        return self.result.__repr__()

    def __repr__(self):
        return self.result.__repr__()
