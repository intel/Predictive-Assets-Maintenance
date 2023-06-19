# Predictive Assets Maintenance

## Introduction

**Reference kit for how to build end to end predictive assets maintenance pipeline.**

Predictive assets maintenance is a proactive approach by leveraging historical data and effectively predicting potential failures to reduce costs, enhance safety and optimize asset performance. Compared with traditional inspection methods, predictive assets maintenance allows organizations to detect and address issues in early stages, preventing unexpected breakdowns and reducing equipment downtime. By identifying potential failures in advance, maintenance can be scheduled strategically to minimize disruptions and ensure continuous operation.

This [workflow](https://github.com/intel/Predictive-Assets-Maintenance/blob/main/AIOps/AIOps_Asset_Maintenance.ipynb) exactly demonstrates how to predict based on historical data and detect checking points (i.e. anomaly) through BigDL Time Series Toolkit (previous known as BigDL-Chronos or Chronos), therefore implement asset maintenance.

Check out more workflow examples [here](https://github.com/intel/Predictive-Assets-Maintenance).

## Solution Technical Overview

We provide an end-to-end (data loading, processing, built-in model, training, inference) AI solution with less operational cost by using BigDL Time Series Toolkit. The BigDL Time Series Toolkit supports building AI solution on single node or cluster and provides more than 10 built-in models for forecasting and anomaly detection. By deploying a model for predicting based on input data and another model for detecting the predicted data, we can effectively detect the potential abnormal behaviors or failures before any significant assets damage occurs.

This workflow shows how to use `TSDataset` to process time series data, `TCNForecaster` to predict future data, `ThresholdDetector` to detect anomalies and therefore implement predictive assets maintenance.

## Solution Technical Details

Specifically, users could load their historical time series data (datetime column is necessary) to [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) for preprocessing (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering (e.g. datetime feature, aggregation feature). And `TSDataset` can be initialized from pandas dataframe, path of parquet file and Prometheus data. It's recommended that users select proper data processing techniques based on characteristics of their dataset to implement better performance. The quality of input data will undoubtedly influence the results of training. Details about how to preprocess data can be found [here](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/how_to_preprocess_my_data.html).


The processed data could be fed into [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) for training and prediction. During initialization of `TCNForecaster`, users can specify the length of historical data, length of predicted data, number of variables to observe and number of variables to predict by setting parameter `past_seq_len` `future_seq_len` `input_feature_num` `output_feature_num` correspondingly.

Then input the predicted data and real data (real data is optional) to [ThresholdDetector](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector) to generate anomaly time points. With these checking points, we can identify potential failures in advance and avoid asset loss in time.

The whole process is shown as following:
![Workflow](https://user-images.githubusercontent.com/108676127/234173176-28ac2046-b215-46ce-b14e-6e3a0e5dd695.jpg)

Provided example workflow shows how to implement predictive maintenance in the elevator industry. In detail, process history data of elevator system, predict the absolute value of vibration and detect the potential failure behaviors.

Users can refer to above documentation and this example to implement predictive assets maintenance with their customized datasets.

## Validated Hardware Details

BigDL Time Series Toolkit and the workflow example shown below could be run widely on both Core™ and Xeon® series processers.

|| Recommended Hardware         | Precision |
|---| ---------------------------- | --------- |
|CPU| Intel® 4th Gen Xeon® Scalable Performance processors| BF16 |
|CPU| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32/INT8 |

### On Premise

#### Training

|| Recommended Hardware         | Precision |
|---| ---------------------------- | --------- |
|CPU| Intel® 4th Gen Xeon® Scalable Performance processors| BF16 |
|CPU| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32/INT8 |

#### Inference

The hardware details of inference process is similiar to training.

## How it Works

Users first load their historical time series data (datetime column is necessary) to [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) for preprocessing and feature engineering. The processed data could be fed into [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) for training and prediction. Then input the predicted data and real data (real data is optional) to [ThresholdDetector](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector) to generate anomaly time points. With these checking points, we can identify potential failures in advance and avoid asset loss in time.

The whole process is shown as following:
![Workflow](https://user-images.githubusercontent.com/108676127/234173176-28ac2046-b215-46ce-b14e-6e3a0e5dd695.jpg)

More details can be found in [Solution Technical Details](#solution-technical-details)

**Note**: To realize satisfactory forecasting perfomance, users may need to select proper data processing techniques of [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) based on characteristics of their customized dataset. And parameters of [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) may also need to tuned for their customized datasets.

## Get Started

Define an environment variable that will store the workspace path, this can be an existing directory or one to be created in further steps. This ENVVAR will be used for all the commands executed using absolute paths.

```
export WORKSPACE=your_workspace_dir
```

### Download the Workflow Repository

Create a working directory for the workflow and all components of the workflow can be found [here](https://github.com/intel/Predictive-Assets-Maintenance/blob/main/AIOps).

```
# For example...
mkdir -p $WORKSPACE && cd $WORKSPACE
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd $WORKSPACE/<workflow repo name>/AIOps
```

### Download the Datasets

The dataset we will use in our workflow example is [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset), which is not contained in our docker image or workflow directory. The dataset contains operation data in the elevator industry, including lots of electromechanical sensors, ambiance and physics data.

Before runnning the workflow script, you need to download [dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset) from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`.

```bash
export DATASET_DIR=your_directory_of_csv_file
cd $DATASET_DIR
# download and decompress to get csv file called `predictive-maintenance-dataset.csv`
cd $WORKSPACE
```

## Supported Runtime Environment
You can execute the references pipelines using the following environments:
* Docker
* Argo 
* Bare metal. 

### Run Using Docker

Follow these instructions to set up and run our provided Docker image. For running on bare metal, see the [bare metal](#run-using-bare-metal) instructions.

#### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

#### Setup Docker Compose
Ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).

```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

#### Set Up Docker Image
Ensure some environment variables are set before running workflow.

```bash
export DATASET_DIR=your_directory_of_csv_file
export FINAL_IMAGE_NAME=workflow # You may choose any image name you want
export http_proxy=your_http_proxy 
export https_proxy=your_https_proxy
```

Then build the provided docker image.

```bash 
cd $WORKSPACE/<workflow repo name>/docker-compose 
docker compose up --build
 ```

#### Run Pipeline with Docker Compose

The container flow diagram using Mermaid is as following:
![Mermaid diagram](https://github.com/plusbang/users-image/assets/108676127/e8c3d893-cc85-4583-84ca-8e79eaa93d7e)

Run entire pipeline:

```bash
docker compose run <service> &
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| DATASET_DIR | `$DATASET_DIR` | Preprocessed Dataset |

#### View Logs
Follow logs of each individual pipeline step using the commands below:

```bash
docker compose logs <service> -f
```

#### Run One workflow with Docker Compose

The Mermaid diagram for workflow is shown as following:
![Mermaid diagram](https://github.com/plusbang/users-image/assets/108676127/e8c3d893-cc85-4583-84ca-8e79eaa93d7e)

Create your own script and run your changes inside of the container using compose as follows:

```bash
cd $WORKSPACE/<workflow repo name>/docker-compose
docker compose run dev
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| DATASET_DIR | `$DATASET_DIR` | Preprocessed Dataset |
| SCRIPT | `src/own-script.sh` | Name of Script |

#### Run Docker Image in an Interactive Environment

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
Run the workflow using the ``docker run`` command, as shown:

```bash
export DATASET_DIR=your_directory_of_csv_file
docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --volume ${DATASET_DIR}:/dataset \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm \
  <workflow image name>
```

#### Clean Up Docker Containers
Stop containers created by docker compose and remove them.

```bash
docker compose down
```

### Run Using Argo
#### 1. Install Helm
If you don't have this tool installed, consult the official [Installing Helm](https://helm.sh/docs/intro/install/).
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```

#### 2. Setting up K8s
Before running workflow on K8s, we need to set up docker image and working directory (i.e. hostpath).
```bash
cd $WORKSPACE/<workflow repo name>/helm-charts/docker
bash build-docker-image.sh # NAME:TAG of the image is intelanalytics/bigdl:asset-maintenance-ubuntu-20.04
```

- Install [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/) and [Argo CLI](https://github.com/argoproj/argo-workflows/releases)
- Configure your [Artifact Repository](https://argoproj.github.io/argo-workflows/configure-artifact-repository/)
- Ensure that your dataset and config files are present in your chosen artifact repository.

#### 3. Install Workflow Template
Prepare and update workflow values in `values.yml`
```bash
cd $WORKSPACE/<workflow repo name>/helm-charts/kubernetes
```
Run this workflow on K8s using helm.
```bash
cd helm-charts/kubernetes
helm install asset-maintenance ./
```

#### 4. View 
To view your workflow progress:
```bash
kubectl get pods
kubectl logs your_pod_name
```

### Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development system. For running a provided Docker image with Docker, see the [Docker instructions](#run-using-docker).


#### Set Up System Software
Our examples use the ``conda`` package and enviroment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

#### Set Up Workflow
Run these commands to set up the workflow's conda environment and install required software:
```
cd $WORKSPACE
git clone https://github.com/intel/Predictive-Assets-Maintenance.git
cd $WORKSPACE/<workflow repo name>/AIOps

conda create -n my_env python=3.9 setuptools=58.0.4
conda activate my_env
pip install --pre --upgrade bigdl-chronos[pytorch] matplotlib notebook==6.4.12
```

Besides, download [dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset) from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`.
Run the following command to download and extract dataset.
```bash
export DATASET_DIR=your_directory_of_csv_file
```

#### Run Workflow
Now we can use these commands to run the workflow:
```bash
# transform the notebook to a runnable python script
jupyter nbconvert --to python AIOps_Asset_Maintenance.ipynb
sed -i '/get_ipython()/d' AIOps_Asset_Maintenance.py
sed -i "s#/dataset/predictive-maintenance-dataset.csv#$DATASET_DIR/predictive-maintenance-dataset.csv#g" AIOps_Asset_Maintenance.py

python AIOps_Asset_Maintenance.py
```

### Expected Output
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
For more information about <workflow> or to read about other relevant workflow
examples, see these guides and software resources:

- [Document First page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/index.html) (you may start from here and navigate to other pages)
- [How to Guides](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/index.html) (If you are meeting with some specific problems during the usage, how-to guides are good place to be checked.)
- [Example & Use-case library](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/index.html) (Examples provides short, high quality use case that users can emulated in their own works.)

## Troubleshooting
Nothing for now.

## Support
If you have any questions with this workflow, the team tracks both bugs and enhancement requests using [GitHub
issues](https://github.com/intel/Predictive-Assets-Maintenance/issues).
