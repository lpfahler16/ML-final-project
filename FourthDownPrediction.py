import importlib.util
from joblib import load
import os
import torch

scripts_path = os.path.abspath('../scripts')
spec = importlib.util.spec_from_file_location(
    "nn_models", os.path.join(scripts_path, "nn_models.py"))
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class FourthDownPrediction:

    def __init__(self, play_model='default', conversion_by_nn=False):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if play_model == 'default':
            play_model = 'RandomForestCV'

        if play_model == 'RandomForest':
            self.attempt_model = load(os.path.join(
                current_dir, './models/random_forest/attempt.joblib'))
            self.convert_model = load(os.path.join(
                current_dir, './models/random_forest/convert.joblib'))
            self.sklearn = True
        elif play_model == 'RandomForestCV':
            self.attempt_model = load(os.path.join(
                current_dir, './models/random_forest/attempt_cv.joblib'))
            self.convert_model = load(os.path.join(
                current_dir, './models/random_forest/convert_cv.joblib'))
            self.sklearn = True
        elif play_model == 'KNN':
            self.attempt_model = load(os.path.join(
                current_dir, './models/knn/attempt.joblib'))
            self.convert_model = load(os.path.join(
                current_dir, './models/knn/convert.joblib'))
            self.sklearn = True
        elif play_model == 'KNNCV':
            self.attempt_model = load(os.path.join(
                current_dir, './models/knn/attempt_cv.joblib'))
            self.convert_model = load(os.path.join(
                current_dir, './models/knn/convert_cv.joblib'))
            self.sklearn = True
        elif play_model == 'NeuralNetwork':
            self.attempt_model = models.AttemptNNClassifier()
            self.attempt_model.load_state_dict(torch.load(os.path.join(
                current_dir, './models/nn/attempt.joblib')))
            self.convert_model = models.ConvertNNClassifier()
            self.convert_model.load_state_dict(torch.load(os.path.join(
                current_dir, './models/nn/convert.joblib')))
            self.sklearn = False
        else:
            raise ValueError('Invalid play model given')

        self.scaler = load(os.path.join(current_dir, './models/scaler.joblib'))
        self.conversion_by_nn = conversion_by_nn

        self.conversion_model = models.ConversionNNClassifier()
        self.conversion_model.load_state_dict(torch.load(os.path.join(
            current_dir, './models/nn/conversion.joblib')))

    def predict_play(self, values, include_conversion=False, scale=False):
        if scale:
            values = self.scaler.transform(values)

        model = self.convert_model if include_conversion else self.attempt_model

        if self.sklearn:
            return model.predict(values)
        else:
            output = model(torch.tensor(values).float())
            preds = output.argmax(dim=1, keepdim=True)
            mapping = {
                0: 'CONVERTED',
                1: 'FAILED',
                2: 'FIELD_GOAL',
                3: 'PUNT'
            } if include_conversion else {
                0: 'ATTEMPTED',
                1: 'FIELD_GOAL',
                2: 'PUNT'
            }
            return [mapping[n.item()] for n in preds]

    def predict_conversion(self, values, percentage=False, scale=False):
        if scale:
            values = self.scaler.transform(values)

        output = self.conversion_model(torch.tensor(values).float())
        preds = output.argmax(dim=1, keepdim=True)
        return preds
