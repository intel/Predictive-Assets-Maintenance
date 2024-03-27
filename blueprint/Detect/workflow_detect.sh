#!/bin/bash

cp -r $INPUT_FORECAST/output/ /cnvrg/
rm -rf /workspace/output
ln -s /cnvrg/output /workspace/
ln -s /cnvrg/anomaly-detect.py /workspace/anomaly-detect.py
cd /workspace
python anomaly-detect.py