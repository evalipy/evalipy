import pickle
from joblib import load


class Model:
    def __init__(self, model, model_type=None, path=None) -> None:
        self.path = path
        self.model = self.load_model() if model is None else model
        self.model_type = type(self.model)
        self.parameters = self.model.get_params()
        pass

    def __repr__(self) -> str:
        return (f"path:{str(self.path)} "
                f"type:{str(self.model_type)} "
                f"parameters:{str(self.parameters)}"
                )
        pass

    def load_model(self) -> object:
        try:
            model = pickle.load(self.path)
            return model
        except Exception as ex:
            print(ex)
