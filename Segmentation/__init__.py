"""
    Recommended use:
        from Segmentation import model_manager
        model = model_manager.load_model("my_favorite_model")
"""

import os

from model_manager import ModelManager as _ModelManager

model_dir = os.path.join(
    os.path.dirname(__file__),
    "models"
)
model_manager = _ModelManager(model_dir)
