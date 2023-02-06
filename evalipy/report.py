import pandas as pd
import numpy as np
from .metrics import RegressionMetrics, ClassificationMetrics
from .model import Model


class Report:
    """
    Report class is used to generate reports currently based on metrics module.
    """

    def __init__(self, model: Model, actual_data: np.ndarray, predicted_data: np.ndarray, x=None,
                 model_identifier='model'):
        self.model = model
        self.actual_data = actual_data
        self.predicted_data = predicted_data if (predicted_data is not None) else model.model.predict(x)
        self.model_identifier = model_identifier
        self.report_DataFrame = self.__generate_report()

    def __generate_report(self):
        kind = self.model.raw_type.lower().find("regress")
        match kind:
            case -1:
                return pd.DataFrame.from_dict(
                    {
                        x: ClassificationMetrics.ALL_CLASSIFICATION_METRICS[x](self.actual_data, self.predicted_data)
                        for x in
                        ClassificationMetrics.ALL_CLASSIFICATION_METRICS.keys()

                    }, orient='index', columns=[f'{self.model_identifier}'])
            case _:
                return pd.DataFrame.from_dict(
                    {
                        x: RegressionMetrics.ALL_REGRESSION_METRICS[x](self.actual_data, self.predicted_data) for x in
                        RegressionMetrics.ALL_REGRESSION_METRICS.keys()

                    }, orient='index', columns=[f'{self.model_identifier}'])

    def __str__(self) -> str:
        return self.report_DataFrame.__repr__()

    def __repr__(self) -> str:
        return self.report_DataFrame.__repr__()
