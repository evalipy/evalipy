import pandas as pd
from numpy import ndarray
from .model import Model
from .report import Report


class Comparator:
    def __init__(self, models: list[Model], x: ndarray, actual_data: ndarray) -> None:
        self.models = [Model.to_model(i) for i in models]
        self.actual_data = actual_data
        self.x = x
        self.result = self.__res()

    def __res(self):
        return pd.concat(
            [Report(model=i, actual_data=self.actual_data, predicted_data=i.model.predict(self.x),
                    model_identifier=f"({self.models.index(i)}) {i.model_type}").report_DataFrame for i in self.models
             ],
            axis=1,
            ignore_index=False)

    def __str__(self):
        return self.result.__repr__()

    def __repr__(self):
        return self.result.__repr__()
