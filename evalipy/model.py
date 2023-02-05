import pickle
from joblib import load


class Model:
    def __init__(self, model, model_type=None, path=None) -> None:
        self.path = path
        self.model = self.load_model() if model is None else model
        self.model_type = model_type if model_type is not None else str(type(self.model)).replace("'>", '').split('.')[
            -1]
        self.raw_type = str(type(self.model))
        self.parameters = self.model.get_params()

        pass

    @staticmethod
    def to_model(model):
        if type(model) is Model:
            return model
        return Model(model)

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
