import os
if 'https_proxy' in os.environ.keys():
    os.environ['KAGGLE_PROXY'] = os.environ['https_proxy']
elif 'HTTPS_PROXY' in os.environ.keys():
    os.environ['KAGGLE_PROXY'] = os.environ['HTTPS_PROXY']
else:
    os.environ['KAGGLE_PROXY'] = ''

import json
data = { 'username' : os.environ['KAGGLE_USERNAME'], 'key': os.environ['KAGGLE_KEY'] }

# Create empty kaggle config and make it owned by user only
kaggle_config = ('kaggle.json')
with open(kaggle_config, 'w') as f:
    os.chmod(kaggle_config, 0o600)
    pass

# Write the config to the file
with open(kaggle_config, 'w') as f:
    json.dump(data, f)

# Download dataset
import opendatasets as od
dataset_url = 'https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset'
od.download(dataset_url, force=True)

# Delete kaggle config
os.remove(kaggle_config)
