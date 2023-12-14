import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

"""
Converts a parquet file to a pandas DataFrame.

Args:
    path (str): The path to the parquet file.

Returns:
    pandas.DataFrame: The converted DataFrame.

Examples:
    >>> convert_df('data.parquet')
    datetime
    2022-01-01 00:00:00    10.0
    2022-01-01 01:00:00    15.0
    2022-01-01 02:00:00    20.0
    ...
"""
def convert_df(path):
    dframe = pd.read_parquet(path)
    dframe.index = pd.to_datetime(dframe['datumEindeMeting'])
    dframe.index.name = None
    dframe.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
    dframe = dframe["hstWaarde"].astype(float)
    dframe = dframe[:-1]
    return dframe

ammonium_df = convert_df("../../data/tank1/ammonium.parquet")
nitrate_df = convert_df("../../data/tank1/nitrate.parquet")
phosphate_df = convert_df("../../data/tank1/phosphate.parquet")
oxygen_df = convert_df("../../data/tank1/oxygen_a.parquet")
energy_df = convert_df("../../data/tank1/energy.parquet")
water_df = pd.read_csv("../../data/tank1/water.csv", delimiter=";")

water_df.index = pd.to_datetime(water_df['DateTime'], format='%d-%m-%Y %H:%M')
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].str.replace(',', '.').replace('(null)', np.nan, regex=True).astype(float)
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].interpolate()
water_df = water_df['EDE_09902MTW_K100.MTW']
water_df.index.name = None
water_df = water_df[water_df.index.isin(oxygen_df.index)]
water_df = water_df[~water_df.index.duplicated()]

ammonium_df = ammonium_df[ammonium_df.index.isin(water_df.index)]
nitrate_df = nitrate_df[nitrate_df.index.isin(water_df.index)]
phosphate_df = phosphate_df[phosphate_df.index.isin(water_df.index)]
oxygen_df = oxygen_df[oxygen_df.index.isin(water_df.index)]
energy_df = energy_df[energy_df.index.isin(water_df.index)]

# oxygen_binary = oxygen_df >= 1

print(phosphate_df, ammonium_df, nitrate_df, energy_df, water_df, oxygen_df)

merged_df = pd.concat([phosphate_df, 
                       ammonium_df, 
                       nitrate_df, 
                       energy_df,
                       water_df,
                       oxygen_df], axis=1)

# Renaming columns for clarity
merged_df.columns = ['Phosphate', 'Ammonium', 'Nitrate', 'Energy', 'WaterFlow', 'Oxygen']

# Handling missing values - filling with mean (simple approach, can be improved)
merged_df.fillna(merged_df.mean(), inplace=True)

# Feature Engineering: Adding time-related features (e.g., hour of the day, day of the week)
merged_df['DateTime'] = pd.to_datetime(oxygen_df.index)
merged_df['HourOfDay'] = merged_df['DateTime'].dt.hour
merged_df['DayOfWeek'] = merged_df['DateTime'].dt.dayofweek
merged_df['MonthOfYear'] = merged_df['DateTime'].dt.month


# Dropping the original datetime column
merged_df.drop('DateTime', axis=1, inplace=True)

# Splitting the dataset into features (X) and target variable (y)
X = merged_df.drop('Oxygen', axis=1)
y = merged_df['Oxygen']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a RandomForest model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print(f"Training Set - Mean Squared Error: {mse_train}, R-squared: {r2_train}")
print(f"Testing Set - Mean Squared Error: {mse}, R-squared: {r2}")

# Visualizing the feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

plt.show()

print(mse, r2)

# Creating a DataFrame from y_test and y_pred for alignment
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

joblib.dump(results_df, "results.joblib")

# Sorting by index (time) to maintain time series order
results_df = results_df.sort_index()

# Plotting the time series
plt.figure(figsize=(10, 6))
plt.plot(results_df['Actual'], label='Actual Oxygen Levels', alpha=0.7)
plt.plot(results_df['Predicted'], label='Predicted Oxygen Levels', alpha=0.7)
plt.xlabel('Timestamp')
plt.ylabel('Oxygen Levels')
plt.title('Time Series of Actual vs Predicted Oxygen Levels')
plt.legend()
plt.show()

