
from ast import FunctionType, literal_eval
import json
import marshal
import os
import types
from typing import Any

import torch

CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_DIR = CURRENT_DIR + os.path.sep + "Models"
DEFAULT_MODEL_INFO = "ModelInfo.json"
MODEL_INFO_KEY_INITIALIZER = "initializer"
MODEL_INFO_INTIALIZER_STRING_PREFIX = "DO NOT TOUCH THIS STRING AT ALL --> "

class ModelManager(object):

    # TODO - redo the process of saving the model initializer

    def __init__(self, dir_path: str = DEFAULT_DIR, model_info_filename: str = DEFAULT_MODEL_INFO):

        self.dir_path = dir_path
        self.model_info_path = self.dir_path + os.path.sep + model_info_filename

        assert (os.path.isfile(self.model_info_path))

        with open(self.model_info_path) as f:
            self.model_info = json.load(f)

    def save_model(self, model: torch.nn.Module, model_name: str, model_info: dict, overwrite_model: bool = False):

        if not overwrite_model and model_name in self.model_info:
            raise Exception(f"can't overwrite model {model_name}")

        self.validate_model_info(model_info)

        model_info[MODEL_INFO_KEY_INITIALIZER] = self.convert_model_info_initializer_to_string_representation(model_info[MODEL_INFO_KEY_INITIALIZER])
        self.model_info[model_name] = model_info
        torch.save(model.state_dict(), self.make_model_path(model_name))

        self.save_model_info()

    def load_model(self, model_name: str) -> torch.nn.Module:

        if model_name not in self.model_info:
            raise Exception(f"{model_name} not in model_info")
        
        model_info = self.model_info[model_name]
        model_initializer = self.convert_model_info_intializer_from_string_representation(model_info[MODEL_INFO_KEY_INITIALIZER], model_name)
        model = model_initializer()
        model.load_state_dict(torch.load(self.make_model_path(model_name)))
        model.eval()
        return model

    def validate_model_info(self, model_info) -> bool:

        if MODEL_INFO_KEY_INITIALIZER not in model_info:
            raise Exception(f"model_info needs to contain {MODEL_INFO_KEY_INITIALIZER}, a function for initializing the model")
        if not ModelManager.is_json_serializable({k:v for k,v in model_info.items() if k != MODEL_INFO_KEY_INITIALIZER}):
            raise Exception("model is not json serializable")

    def convert_model_info_initializer_to_string_representation(self, initializer: FunctionType):

        # TODO - make this process better? enforce no args?
        return MODEL_INFO_INTIALIZER_STRING_PREFIX + str(marshal.dumps(initializer.__code__))

    def convert_model_info_intializer_from_string_representation(self, initializer_string_representation, model_name):

        # TODO - make this process better?
        initializer_string_representation = initializer_string_representation.replace(MODEL_INFO_INTIALIZER_STRING_PREFIX, "")
        initializer = types.FunctionType(marshal.loads(literal_eval(initializer_string_representation)), globals(), f"{model_name}_initializer_from_model_info")
        return initializer

    def make_model_path(self, model_name: str):

        return f"{self.dir_path}{os.path.sep}{model_name}.model"

    def save_model_info(self):

        if not ModelManager.is_json_serializable(self.model_info):
            raise Exception("Ruh roh")
        with open(self.model_info_path, 'w') as f:
            json.dump(self.model_info, f)
    
    @staticmethod
    def is_json_serializable(data: Any) -> bool:

        try:
            str = json.dumps(data)
            return True
        except Exception as e:
            return False

    def __del__(self):

        self.save_model_info()
