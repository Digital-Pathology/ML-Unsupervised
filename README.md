# Cell Level Models

## Workflow Overview

The unsupervised models consist of a few components:
- Segmentation
- Feature Engineering
- Classification (unsupervised)

A segmentation model segments the cells or cell components of note from the image in question. The cells are then reduced to a vector of features according to a feature manager. The feature representations of the cells are then classified by a classification model.

Each component can be developed independently, and a pipeline was established to facilitate hotswapping various implementations for each component.

## Segmentation

TODO

## Feature Engineering

TODO

## Classification

TODO
