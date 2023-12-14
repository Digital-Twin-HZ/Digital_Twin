import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split

def convert_df(path):
    dframe = pd.read_parquet(path)
    dframe.index = pd.to_datetime(dframe['datumBeginMeting'])
    dframe.index.name = None
    dframe.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
    dframe = dframe["hstWaarde"].astype(float)
    return dframe

# PREDICTIONS
# init = joblib.load("predictions.joblib")
# np.savetxt('test2.txt', init, delimiter=',')

pred = np.loadtxt('test2.txt', dtype=float)

# The model spits out negative values, clipping these to 0 provides us with the periods when the blowers were off.
# TODO: investigate why it spits negative values (there are no features provided with below 0 values)
predictions = np.clip(pred, 0, None)
# np.savetxt('pred.txt', predictions, delimiter=',')

# LOAD ORIGINAL
ox = convert_df("../data/tank1_new/oxygen_a.parquet")

percent = 20
n_rows_to_select = int(len(ox) * percent / 100)

# Use tail to select the last 20% of the time series
last_20_percent = ox.tail(n_rows_to_select)

predictions = pd.Series(predictions[1:], index=last_20_percent.index)

# MEAN AVERAGE ERROR
ae = np.absolute((last_20_percent - predictions))
mae = np.sum(ae) / len(ae)
print(mae)

'''
These values are fantastic to select a certain period of the entirety of the data.
'''
target_month = 11
target_day = 20
target_hour = 8
last_20_percent = last_20_percent[(last_20_percent.index.month == target_month) & (last_20_percent.index.day == target_day)]
predictions = predictions[(predictions.index.month == target_month) & (predictions.index.day == target_day)]

plt.figure(figsize=(8, 6))
plt.plot(last_20_percent, label='ACTUAL')
plt.plot(predictions, label='PRED')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ACTUAL VS PRED')
plt.legend()
plt.grid(True)
plt.show()