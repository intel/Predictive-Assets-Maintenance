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
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.pytorch.loss import AsymWeightLoss
from bigdl.chronos.metric.forecast_metrics import Evaluator

lookback = 120
horizon = 1

df = pd.read_csv("predictive-maintenance-dataset.csv")
df["time_step"] = pd.date_range(start='2023-01-01 16:30:00', end='2023-01-01 23:30:00', periods=len(df))

# Feature Engineering & Data Preperation
tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="time_step", target_col="vibration",
                                                              extra_feature_col=["revolutions","humidity","x1","x2","x3","x4","x5"],
                                                              with_split=True, test_ratio=0.1)
standard_scaler = StandardScaler()

for tsdata in [tsdata_train, tsdata_test]:
    tsdata.scale(standard_scaler, fit=(tsdata is tsdata_train))\
          .roll(lookback=lookback, horizon=horizon)

x_train, y_train = tsdata_train.to_numpy()
x_test, y_test = tsdata_test.to_numpy()

# Time series forecasting
forecaster = TCNForecaster(past_seq_len=lookback,
                           future_seq_len=horizon,
                           input_feature_num=8,
                           output_feature_num=1,
                           normalization=False,
                           kernel_size=5,
                           num_channels=[16]*8,
                           loss=AsymWeightLoss(underestimation_penalty=10))

print('Start training ...')
forecaster.num_processes = 1
forecaster.fit(data=tsdata_train, epochs=5)
print('Training completed')

x_train, y_train = tsdata_train.to_numpy()
x_test, y_test = tsdata_test.to_numpy()

y_pred_train = forecaster.predict(x_train)
y_pred_test = forecaster.predict(x_test)

y_pred_train_unscale = tsdata_train.unscale_numpy(y_pred_train)
y_pred_test_unscale = tsdata_test.unscale_numpy(y_pred_test)
y_train_unscale = tsdata_train.unscale_numpy(y_train)
y_test_unscale = tsdata_test.unscale_numpy(y_test)
metric = Evaluator.evaluate('mse', y_test_unscale, y_pred_test_unscale)
print(f"MSE is {'%.2f' % metric[0]}")

np.save('output/y_pred_train_unscale.npy', y_pred_train_unscale)
np.save('output/y_pred_test_unscale.npy', y_pred_test_unscale)
np.save('output/y_train_unscale.npy', y_train_unscale)
np.save('output/y_test_unscale.npy', y_test_unscale)
