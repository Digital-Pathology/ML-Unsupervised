#!/bin/bash

copy .
cd /workspaces/dev-container/SagemakerTemplate/code
rm -rf \
    .git \
    databases \
    testing
build_and_push adin_model_eval_test
