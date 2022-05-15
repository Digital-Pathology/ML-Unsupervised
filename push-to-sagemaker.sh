#!/bin/bash

IMAGE_NAME=adin_tiling

date

# copy material over from this directory
time copy .

# remove the big unnecessary files
cd /workspaces/dev-container/SagemakerTemplate/code
rm -rf \
    .devcontainer \
    .git* \
    archive \
    datasets \
    testing \
    *.ipynb \
    filtration_cache.h5 \
    README.md \
    model_analysis_results_weight_ratio.csv \
    region_predictions_aggregated.json \
    output \
    sagemaker-template.sh \
    environment.yml

# do the thing
time build_and_push $IMAGE_NAME

date

echo "pushed to" $IMAGE_NAME
