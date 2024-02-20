#!/bin/bash

mkdir /cnvrg/output
rm -rf /workspace/output
rm -f /workspace/*
cp -r $INPUT_DOWNLOAD_DATASET/elevator-predictive-maintenance-dataset /cnvrg/
ln -s /cnvrg/output /workspace/
ln -s /cnvrg/elevator-predictive-maintenance-dataset/predictive-maintenance-dataset.csv /workspace/predictive-maintenance-dataset.csv
ln -s /cnvrg/forecast.py /workspace/forecast.py
cd /workspace
export OMP_NUM_THREADS=2
python forecast.py
rm -rf /cnvrg/elevator-predictive-maintenance-dataset
