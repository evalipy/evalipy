import pandas as pd
from report import Report


class Comparator:
    def __init__(self, model_A, model_B, actual_data, predicted_data_A, predicted_data_B) -> None:
        self.model_A = model_A
        self.model_B = model_B
        self.actual_data = actual_data
        self.predicted_data_A = predicted_data_A
        self.predicted_data_B = predicted_data_B

    def __repr__(self):
        return pd.concat([Report(self.model_A, self.actual_data, self.predicted_data_A).__repr__(),
                   Report(self.model_B, self.actual_data, self.predicted_data_B).__repr__()], axis=0, ignore_index=True)
