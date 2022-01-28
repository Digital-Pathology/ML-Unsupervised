# ML-Unsupervised

## Workflow Overview

The unsupervised models consist of a few components:
- Segmentation
- Feature Engineering
- Classification (unsupervised)

A segmentation model segments the cells or cell components of note from the image in question. The cells are then reduced to a vector of features according to a feature manager. The feature representations of the cells are then classified by a classification model.

Each component can be developed independently, and a pipeline was established to facilitate hotswapping various implementations for each component.

## Segmentation

The segmentation module exists to keep various pytorch models accessible for testing and deployment purposes. It has a single interface `ModelManager`. To use the segmentation models import like so:
```python
from Segmentation import ModelManager
model = ModelManager.load_model("my_favorite_model")
```

Saving a model is a bit complicated because of the potential for dependencies. To ensure that the model can be deployed successfully, the user who saves the model using `ModelManager.save_model(model, model_name, model_info)` needs to specify a function that initializes the model. Any dependencies should be imported in that initialization function. Please direct any and all questions to Adin at adinbsolomon@gmail.com.

## Feature Engineering

TODO

## Classification

TODO
