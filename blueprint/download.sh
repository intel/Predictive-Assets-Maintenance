#!/bin/bash

pip install --no-cache-dir opendatasets
ln -s /cnvrg/output /workspace/
ln -s /cnvrg/download.py /workspace/download.py
cd /workspace
python download.py
cp -r /workspace/elevator-predictive-maintenance-dataset /cnvrg