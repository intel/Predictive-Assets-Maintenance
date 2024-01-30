#!/usr/bin/env python
# coding: utf-8

# # Predictive Assets Maintenance with BigDL Time Series Toolkit

# In this notebook we demonstrate how to use `TCNForecaster` and `ThresholdDetector` to realize prediction, anomaly detection and therefore assets maintenance.

# For demonstration, we use the publicly available elevator predictive maintenance dataset. You can find the dataset introduction <a href="https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset" target="_blank">here</a>. The targe is to predict absolute value of vibration. then maintenance teams can be alerted to inspect and address potential issue proactively.

# Before runnning the notebook, you need to download <a href="https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset" target="_blank">dataset</a> from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`.

# ## Helper function

# This section provides a helper function to plot the ground truth, prediction and anomaly value. You can refer to it later when use.

# In[1]:


def plot_anomalies_value(y_true, y_pred, pattern_ano_index, trend_ano_index, threshold):
    """
    Plot the ground truth, prediction and anomaly value.
    """
    df = pd.DataFrame({"y_true": y_true.squeeze(), "y_pred": y_pred.squeeze()})
    df['p_ano_index'] = 0
    df.loc[df.index[pattern_ano_index], 'ano_index'] = 1
    df['t_ano_index'] = 0
    df.loc[df.index[trend_ano_index], 'ano_index'] = 1
    df['threshold'] = threshold

    fig, axs = plt.subplots(figsize=(16,6))
    axs.plot(df.index, df.y_true, color='blue', label='Ground Truth')
    axs.plot(df.index, df.y_pred, color='orange', label='Prediction')
    axs.plot(df.index, df.threshold, color='black', label='Threshold')
    axs.scatter(df.index[pattern_ano_index].tolist(), df.y_true[pattern_ano_index], color='red', label='checking points for pattern anomaly')
    axs.scatter(df.index[trend_ano_index].tolist(), df.y_true[trend_ano_index], color='green', label='checking points for trend anomaly')
    axs.set_title('Checking Points For Maintenance')
    
    plt.xlabel('time_step')
    plt.legend(loc='upper left')
    plt.show()


# ## Download raw dataset and load into dataframe

# Download <a href="https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset" target="_blank">dataset</a> from kaggle and decompress it to get csv file called `predictive-maintenance-dataset.csv`. And use pandas to load `predictive-maintenance-dataset.csv` into a dataframe as shown below.

# In[2]:


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("/dataset/predictive-maintenance-dataset.csv")


# Below are some example records of the data

# In[4]:


df.head()


# In[5]:


df.plot(y="vibration", x="ID", figsize=(16,6), title="vibration")


# ## Data pre-processing

# Now we need to do data cleaning and preprocessing on the raw data. Note that this part and the following part could vary for different dataset. 
# 
# For the elevator data, the pre-processing convert the time step to timestamp starting from 2023-01-01 16:30:00.

# In[6]:


df["time_step"] = pd.date_range(start='2023-01-01 16:30:00', end='2023-01-01 23:30:00', periods=len(df))
df


# ## Feature Engineering & Data Preperation

# We scale and roll the data to generate the sample in numpy ndarray for `TCNForecaster` to use.
# 
# We use <a href="https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html" target="_blank">TSDataset</a> to complete the whole processing.

# In[7]:


from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler

lookback = 120
horizon = 1

tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="time_step", target_col="vibration",
                                                              extra_feature_col=["revolutions","humidity","x1","x2","x3","x4","x5"],
                                                              with_split=True, test_ratio=0.1)
standard_scaler = StandardScaler()

for tsdata in [tsdata_train, tsdata_test]:
    tsdata.scale(standard_scaler, fit=(tsdata is tsdata_train))\
          .roll(lookback=lookback, horizon=horizon)

x_train, y_train = tsdata_train.to_numpy()
x_test, y_test = tsdata_test.to_numpy()
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ## Time series forecasting

# First, we initialize a TCNForecaster based on time step and feature number. More information about TCNForecaster can be found <a href="https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster" target="_blank">here</a>.

# In some industrial scenarios, such as this one, the adverse effect caused by predicted value being less than real value is far greater than that caused by predicted value being greater than real value. Therefore, in this case, we use a built-in loss function `AsymWeightLoss` to penalize underestimation.

# In[8]:


from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.pytorch.loss import AsymWeightLoss


# In[11]:


forecaster = TCNForecaster(past_seq_len=lookback,
                           future_seq_len=horizon,
                           input_feature_num=8,
                           output_feature_num=1,
                           normalization=False,
                           kernel_size=5,
                           num_channels=[16]*8,
                           loss=AsymWeightLoss(underestimation_penalty=10))


# Now we train the model and wait till it finished.

# In[12]:


print('Start training ...')
forecaster.num_processes = 1
forecaster.fit(data=tsdata_train, epochs=5)
print('Training completed')


# Then we can use the fitted forecaster for prediction and inverse the scaling of the prediction results. 

# In[13]:


y_pred_train = forecaster.predict(x_train)
y_pred_test = forecaster.predict(x_test)


# In[14]:


y_pred_train_unscale = tsdata_train.unscale_numpy(y_pred_train)
y_pred_test_unscale = tsdata_test.unscale_numpy(y_pred_test)
y_train_unscale = tsdata_train.unscale_numpy(y_train)
y_test_unscale = tsdata_test.unscale_numpy(y_test)


# In[15]:


from bigdl.chronos.metric.forecast_metrics import Evaluator
metric = Evaluator.evaluate('mse', y_test_unscale, y_pred_test_unscale)
print(f"MSE is {'%.2f' % metric[0]}")


# ## Checking points detection

# Then we initiate a ThresholdDetector to detect checking points, i.e. anomaly need to pay attention. More information about ThresholdDetector can be found <a href="https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector" target="_blank">here</a>. Based on train dataset, we can train it to obtain some information about threshold.

# Moreover, in this case, we can set the absolute threshold of vibration and detect potential elevator failure.

# In[16]:


import math
from bigdl.chronos.detector.anomaly import ThresholdDetector
thd = ThresholdDetector()
vibration_th = 85
thd.set_params(trend_threshold=(0, vibration_th)) # if vibration>85, we think there may exist potential elevator failure
thd.fit(y_train_unscale, y_pred_train_unscale)


# We detect two types of anomaly, i.e. pattern anomaly and trend anomaly. By comparing real data and predicted data, we find those pattern anomalies. Meanwhile, we also support forecasting anomaly, that is detect trend anomaly of predicted data.

# **Case1**: If we only have predicted data and want to forecasting anomaly

# In[17]:


test_anomaly_indexes = thd.anomaly_indexes(y_pred=y_pred_test_unscale)
print("The index of anomalies in test dataset only according to predict data is:")
for key, value in test_anomaly_indexes.items():
    print(f'{key}: {value}')


# Use `plot_anomalies_value` to intuitively feel the detection results.

# In[18]:


plot_anomalies_value(y_test_unscale, y_pred_test_unscale, test_anomaly_indexes['pattern anomaly index'], test_anomaly_indexes['trend anomaly index'], vibration_th)


# **Case2**: If we have true data and predicted data and want to detect anomaly

# In[19]:


test_anomaly_indexes = thd.anomaly_indexes(y=y_test_unscale, y_pred=y_pred_test_unscale)
print("The index of anomalies in test dataset according to true data and predicted data is:")
for key, value in test_anomaly_indexes.items():
    print(f'{key}: {value}')


# Use `plot_anomalies_value` to intuitively feel the detection results.

# In[20]:


plot_anomalies_value(y_test_unscale, y_pred_test_unscale, test_anomaly_indexes['pattern anomaly index'], test_anomaly_indexes['trend anomaly index'], vibration_th)


# Then, we can focus on these checking points to avoid asset loss in time.
