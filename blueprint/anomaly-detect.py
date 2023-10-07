#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pandas as pd
from bigdl.chronos.detector.anomaly import ThresholdDetector
import matplotlib.pyplot as plt

def plot_anomalies_value(y_true, y_pred, pattern_ano_index, trend_ano_index, threshold, fig_name):
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
    plt.savefig(fig_name)
    plt.show()

y_pred_train_unscale = np.load('output/y_pred_train_unscale.npy')
y_pred_test_unscale = np.load('output/y_pred_test_unscale.npy')
y_train_unscale = np.load('output/y_train_unscale.npy')
y_test_unscale = np.load('output/y_test_unscale.npy')

thd = ThresholdDetector()
vibration_th = 85
thd.set_params(trend_threshold=(0, vibration_th)) # if vibration>85, we think there may exist potential elevator failure
thd.fit(y_train_unscale, y_pred_train_unscale)

test_anomaly_indexes = thd.anomaly_indexes(y_pred=y_pred_test_unscale)
print("The index of anomalies in test dataset only according to predict data is:")
for key, value in test_anomaly_indexes.items():
    print(f'{key}: {value}')

plot_anomalies_value(y_test_unscale, y_pred_test_unscale,
                     test_anomaly_indexes['pattern anomaly index'], test_anomaly_indexes['trend anomaly index'],
                     vibration_th, 'output/anomaly_index_only_predicted_data.png')

test_anomaly_indexes = thd.anomaly_indexes(y=y_test_unscale, y_pred=y_pred_test_unscale)
print("The index of anomalies in test dataset according to true data and predicted data is:")
for key, value in test_anomaly_indexes.items():
    print(f'{key}: {value}')

plot_anomalies_value(y_test_unscale, y_pred_test_unscale,
                     test_anomaly_indexes['pattern anomaly index'], test_anomaly_indexes['trend anomaly index'],
                     vibration_th, 'output/anomaly_index_true_and_predicted_data.png')
