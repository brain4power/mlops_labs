#!/bin/sh

pip install requirements
mkdir -p train
mkdir -p test
mkdir -p models
python3 data_creation.py
python3 data_preprocessing.py
python3 model_preparation.py
python3 model_testing.py
