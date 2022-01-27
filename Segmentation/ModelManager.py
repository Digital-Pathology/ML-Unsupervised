
import json
import os
from typing import Any

CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_DIR = CURRENT_DIR + os.path.sep + "Models"
DEFAULT_MODEL_INFO = "ModelInfo.json"

class ModelManager(object):

    def __init__(self, dir_path: str = DEFAULT_DIR, model_info_filename: str = DEFAULT_MODEL_INFO):

        self.dir_path = dir_path
        self.model_info_path = self.dir_path + os.path.sep + model_info_filename

        assert (os.path.isfile(self.model_info_path))

        with open(self.model_info_path) as f:
            self.model_info = json.load(f)

    def save_model(self, model_name, info: Any, overwrite_model: bool = False):

        if not ModelManager.is_json_serializable(info):
            raise Exception("info is not JSON serializable")
        if not overwrite_model and model_name in self.model_info:
            raise Exception(f"can't overwrite model {model_name}")
        self.model_info[model_name] = info
        # TODO - save model to {model_name}.model or something using torch's model saving feature
        raise NotImplementedError()

    def load_model(self, model_name: str):

        if model_name not in self.model_info:
            raise Exception(f"{model_name} not in model_info")
        # TODO - load model using pytorch's model loading feature
        raise NotImplementedError()
    
    @staticmethod
    def is_json_serializable(data: Any) -> bool:

        try:
            str = json.dumps(data)
            return True
        except Exception as e:
            return False

    def save_model_info(self):

        if not ModelManager.is_json_serializable(self.model_info):
            raise Exception("Ruh roh")
        with open(self.model_info_path, 'w') as f:
            json.dump(self.model_info, f)

    def __del__(self):

        self.save_model_info()
