# Predictive Assets Maintenance
**Reference kit for how to build end to end predictive assets maintenance pipeline.**

Predictive assets maintenance is a proactive approach by leveraging historical data and effectively predicting potential failures to reduce costs, enhance safety and optimize asset performance. Compared with traditional inspection methods, predictive assets maintenance allows organizations to detect and address issues in early stages, preventing unexpected breakdowns and reducing equipment downtime. By identifying potential failures in advance, maintenance can be scheduled strategically to minimize disruptions and ensure continuous operation.

This [workflow](https://github.com/intel/Predictive-Assets-Maintenance/blob/main/AIOps/AIOps_Asset_Maintenance.ipynb) exactly demonstrates how to predict based on historical data and detect checking points (i.e. anomaly) through BigDL Time Series Toolkit (previous known as BigDL-Chronos or Chronos), therefore implement asset maintenance.

## Overview

In order to realize assets maintenance with less operational cost, we provide a method composed of forecasting and anomaly detecting by using BigDL Time Series Toolkit. Before any significant assets damage occurs, we detect the potential abnormal behaviors or failures.

BigDL Time Series Toolkit supports building end-to-end (data loading, processing, built-in model, training, tuning, inference) AI solution on single node or cluster and provides more than 10 built-in models for forecasting and anomaly detection.

This workflow shows how to use `TCNForecaster` and `ThresholdDetector` to realize prediction, anomaly detection and therefore predictive assets maintenance.

## Hardware Requirements

BigDL Time Series Toolkit and the workflow example shown below could be run widely on both Core™ and Xeon® series processers.

|| Recommended Hardware         | Precision |
|---| ---------------------------- | --------- |
|CPU| Intel® 4th Gen Xeon® Scalable Performance processors| BF16 |
|CPU| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32/INT8 |

## Software Requirements

Linux OS (Ubuntu 20.04) is used in this reference solution. Make sure the following dependencies are installed.

1. `sudo apt update`
2. Pip/Conda

## How it Works

Specifically, users could load their historical time series data (datetime column is necessary) to [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) for preprocessing (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering (e.g. datetime feature, aggregation feature). TSDataset can be initialized from pandas dataframe, path of parquet file and Prometheus data. The processed data could be fed into [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) for training and prediction. Then input the predicted data and real data (real data is optional) to [ThresholdDetector](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector) to generate anomaly time points. With these checking points, we can identify potential failures in advance and avoid asset loss in time.

The whole process is shown as following:
![Workflow](https://user-images.githubusercontent.com/108676127/234173176-28ac2046-b215-46ce-b14e-6e3a0e5dd695.jpg)

Provided example workflow shows how to implement predictive maintenance in the elevator industry. In detail, process history data of elevator system, predict the absolute value of vibration and detect the potential failure behaviors.

Users can refer to above documentation and this example to implement predictive assets maintenance with their customized datasets.

**Note**: To realize satisfactory forecasting perfomance, users may need to tune parameters of [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) for their customized datasets.

## Get Started

### Download the Workflow Repository

Create a working directory for the workflow and all components of the workflow can be found [here](https://github.com/intel/Predictive-Assets-Maintenance/blob/main/AIOps).

```
# For example...
mkdir ~/work && cd ~/work
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd AIOps
```

### Datasets
The dataset we will use in our workflow example is [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset), which is not contained in our docker image or workflow directory. The dataset contains operation data in the elevator industry, including lots of electromechanical sensors, ambiance and physics data.

Before runnning the workflow script, you need to download [dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset) from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`.

## Ways to run this reference use case
This reference kit offers three options for running the fine-tuning and inference processes:

- Docker
- On K8s Using Helm
- Bare Metal

Details about each of these methods can be found below.

## Run Using Docker
Follow these instructions to set up and run our provided Docker image. For running on bare metal, see the [bare metal](#run-using-bare-metal) instructions.

### 1. Set Up Docker Engine and Docker Compose
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.


To build and run this workload inside a Docker Container, ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).


```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

### 2. Set Environment Variables
Ensure some environment variables are set before running workflow.

```bash
export DATASET_DIR=your_directory_of_csv_file
export FINAL_IMAGE_NAME=workflow # You may choose any image name you want
export http_proxy=your_http_proxy 
export https_proxy=your_https_proxy
```

### 3. Run Workflow with Docker Compose
Run entire workflow to view the logs of running containers. In the first container, download raw data and extract it. Then, in next one, we use BigDL time series toolkit to process the data, forecast and detect.

```bash
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd docker-compose
docker compose up --build
```

#### View Logs
Follow logs using the command below:

```bash
docker compose logs -f
```

### 4. Clean Up Docker Containers
Stop containers created by docker compose and remove them.

```bash
docker compose down
```

## Run on K8s Using Helm

### 1. Install Helm
If you don't have this tool installed, consult the official [Installing Helm](https://helm.sh/docs/intro/install/).
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```

### 2. Set Up Docker Image And Working Directory
Before running workflow on K8s, we need to set up docker image and working directory (i.e. hostpath).
```bash
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd helm-charts/docker
bash build-docker-image.sh # NAME:TAG of the image is intelanalytics/bigdl:asset-maintenance-ubuntu-20.04
```

### 3. Prepare Workflow Values
After setting up docker image and working directory, we need to update these values in `values.yml`.
```bash
cd helm-charts/kubernetes
```

### 4. Install Workflow Template
Now, we can run this workflow on K8s using helm.
```bash
cd helm-charts/kubernetes
helm install asset-maintenance ./
```

#### View Logs
To view your workflow progress:
```bash
kubectl get pods
kubectl logs your_pod_name
```

### 5. Clean Up Helm Release
Delete job created by K8s and remove helm release.

```bash
helm delete asset-maintenance
```

## Run Using Bare Metal
### 1. Prepare Environment
Users are encouraged to use the ``conda`` package and enviroment on local computer. If you don't already have ``conda`` installed, see the [Conda Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).
```bash
conda create -n my_env python=3.9 setuptools=58.0.4
conda activate my_env
pip install --pre --upgrade bigdl-chronos[pytorch] matplotlib notebook==6.4.12
```

### 2. Set Up Working Directory
```bash
# If you have done this in section Workflow Repository
# please skip next 2 commands
mkdir ~/work && cd ~/work
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd AIOps
```

### 3. Preprocess the Dataset
Download [dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset) from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`.
Run the following command to download and extract dataset.
```bash
export DATASET_DIR=your_directory_of_csv_file
```

### 4. Run Workflow
Now we can use these commands to run the workflow:
```bash
# transform the notebook to a runnable python script
jupyter nbconvert --to python AIOps_Asset_Maintenance.ipynb
sed -i '/get_ipython()/d' AIOps_Asset_Maintenance.py
sed -i "s#/dataset/predictive-maintenance-dataset.csv#$DATASET_DIR/predictive-maintenance-dataset.csv#g" AIOps_Asset_Maintenance.py

python AIOps_Asset_Maintenance.py
```

## Expected Output
This workflow provides instructions on how to implement asset maintenance through a Temporal Convolution Neural Network and a Threshold Detector on BigDL Time Series Toolkit using time series dataset as an example.

The main part will be the model training progress bar, finally you can obtain the checking points which exist potential issues and need to pay attention in test dataset.
```
Epoch 4: 100%|███████████████████████████████████████████████████████████████████████| 1814/1814 [01:39<00:00, 18.32it/s, loss=0.311]
Training completed
MSE is 25.46
The index of anoalies in test dataset only according to predict data is:
pattern anomaly index: []
trend anomaly index: [3199, 3159, 3079]
anomaly index: [3079, 3159, 3199]
The index of anoalies in test dataset according to true data and predicted data is:
pattern anomaly index: [279, 679, 1079, 1479, 1879, 2279, 2319, 2679, 2719, 2739, 2751, 2783, 3079, 3119, 3139, 3151, 3183, 3479, 3519, 3551, 3879, 3919, 4279, 4679, 5079, 5479, 5879, 6279]
trend anomaly index: [2959, 3599, 2839, 2719, 3359, 2599, 3239, 3119, 2999, 3639, 2879, 3519, 2759, 3399, 2639, 3279, 3159, 2519, 3039, 3199, 2919, 3559, 2799, 3439, 3319, 2559]
anomaly index: [3079, 6279, 5879, 2319, 2959, 3599, 3479, 279, 2839, 2719, 3359, 679, 3879, 2599, 3239, 3119, 2739, 1079, 4279, 2999, 3639, 3319, 3519, 2751, 2879, 3139, 1479, 4679, 2759, 3399, 2559, 3151, 3919, 2639, 3279, 1879, 5079, 3159, 2519, 2783, 3551, 3039, 2279, 5479, 2919, 3559, 3183, 2799, 3439, 2679, 3199]
```

## Summary and Next Steps
The workflow example shows how to use BigDL Time Series Toolkit to implement predictive assets maintenance. Users may refer to [example notebook](https://github.com/intel/Predictive-Assets-Maintenance/blob/main/AIOps/AIOps_Asset_Maintenance.ipynb) to obtain more details.

## Learn More
- [Document First page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/index.html) (you may start from here and navigate to other pages)
- [How to Guides](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/index.html) (If you are meeting with some specific problems during the usage, how-to guides are good place to be checked.)
- [Example & Use-case library](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/index.html) (Examples provides short, high quality use case that users can emulated in their own works.)


## Troubleshooting
Nothing for now

## Support

If you have any questions with this workflow, the team tracks both bugs and enhancement requests using [GitHub
issues](https://github.com/intel/Predictive-Assets-Maintenance/issues).
