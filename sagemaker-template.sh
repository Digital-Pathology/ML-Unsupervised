#!/bin/bash

date

# copy material over from this directory
time copy .

# remove the big unnecessary files
cd /workspaces/dev-container/SagemakerTemplate/code
rm -rf \
    .devcontainer \
    .git \
    datasets \
    testing \
    *.ipynb \
    filtration_cache.h5 \
    README.md \
    model_analysis_results_weight_ratio.csv \
    region_predictions_aggregated.json

# do the thing
time build_and_push adin_model_eval_test

date
