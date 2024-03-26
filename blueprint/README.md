# Predictive Assets Maintenance Reference Use Case

This blueprint is a one click refkit to provide an end-to-end solution for building deploy predictive assets maintenance.

Predictive assets maintenance is a proactive approach by leveraging historical data and effectively predicting potential failures to reduce costs, enhance safety and optimize asset performance. Compared with traditional inspection methods, predictive assets maintenance allows organizations to detect and address issues in early stages, preventing unexpected breakdowns and reducing equipment downtime. By identifying potential failures in advance, maintenance can be scheduled strategically to minimize disruptions and ensure continuous operation.
 
# Flow
1. Click on `Use Blueprint` button.
2. You will be redirected to your blueprint flow page.
3. Go to the project settings section and update the configuration or leave as default to check on built-in demo. To try with other datasources, please change the first `data task` and update the proper data processing techniques of [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) and parameters of [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster)based on characteristics of their customized dataset.
4. Click on the `Run Flow` button.
5. The system will process time series data, training the forecaster, predict future data and detect anomalies, therefore implement predictive assets maintenance.

Finally you can obtain the checking points which exist potential issues and need to pay attention. Take the default [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset) as example, the expected output may be as following:
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
 
# Solution Technical Overview and Detail

The high-level architecture of the reference use case is shown in the diagram below:

![Workflow](https://user-images.githubusercontent.com/108676127/234173176-28ac2046-b215-46ce-b14e-6e3a0e5dd695.jpg)

We first load historical time series data (datetime column is necessary) to [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) for preprocessing and feature engineering. The processed data could be fed into [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) for training and prediction. Then input the predicted data and real data (real data is optional) to [ThresholdDetector](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector) to generate anomaly time points. With these checking points, we can identify potential failures in advance and avoid asset loss in time.

**Note**: In the blueprint flow, we combine the data preprocessing and forecasting in one step.

**Note**: To realize satisfactory forecasting perfomance, users may need to select proper data processing techniques of [TSDataset](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html#tsdataset) based on characteristics of their customized dataset. And parameters of [TCNForecaster](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster) may also need to tuned for their customized datasets.


# Learn More

To read about other use cases and workflows examples, see these resources:

- [Document First page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/index.html) (you may start from here and navigate to other pages)
- [How to Guides](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/index.html) (If you are meeting with some specific problems during the usage, how-to guides are good place to be checked.)
- [Example & Use-case library](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/index.html) (Examples provides short, high quality use case that users can emulated in their own works.)
 
# Support

If you have any questions, the team tracks both bugs and enhancement requests using [GitHub
issues](https://github.com/intel/Predictive-Assets-Maintenance/issues).
