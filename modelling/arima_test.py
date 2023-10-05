import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import pandas as pd

df = pd.read_parquet("../data/tank1_new/ammonium.parquet")
df.index = pd.to_datetime(df['datumBeginMeting'])
df.index.name = None
df.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
df["hstWaarde"] = df["hstWaarde"].astype('float32')

minutely_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
new_df = pd.DataFrame(index=minutely_index)
merged_df = new_df.join(df, how='left').interpolate(method='time')
df = merged_df

df = df["hstWaarde"]

df2 = df
# df2 = df2.resample('H').mean()
df2 = df2.iloc[:100000]

print(df2)

# Perform a train-test split
train_size = int(len(df2) * 0.8)  # 80% for training, adjust as needed
train, test = df2.iloc[:train_size], df2.iloc[train_size:]

# Fit your model
model = pm.auto_arima(train, seasonal=False)

# make your forecasts
forecasts = model.predict(test.shape[0])  # predict N steps into the future

# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(df2.shape[0])
plt.plot(x[:train_size], train, c='blue')
plt.plot(x[train_size:], forecasts, c='green')
plt.show()
