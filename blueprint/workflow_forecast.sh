#!/bin/bash

mkdir /cnvrg/output
rm -rf /workspace/output
rm -f /workspace/*
ln -s /cnvrg/output /workspace/
ln -s /data/predictive-maintainance/predictive-maintenance-dataset.csv /workspace/predictive-maintenance-dataset.csv
ln -s /cnvrg/forecast.py /workspace/forecast.py
cd /workspace
export OMP_NUM_THREADS=2
python forecast.py
