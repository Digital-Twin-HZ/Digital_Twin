import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

def convert_df(path):
    """
    Converts a Parquet file into a Pandas DataFrame.

    Parameters:
    path (str): The path to the Parquet file.

    Returns:
    pandas.DataFrame: The converted DataFrame.
    """
    dframe = pd.read_parquet(path)
    dframe.index = pd.to_datetime(dframe['datumEindeMeting'])
    dframe.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
    dframe = dframe["hstWaarde"].astype(float)
    return dframe[:-1]

# Loading data
ammonium_df = convert_df("../../data/tank1/ammonium.parquet")
nitrate_df = convert_df("../../data/tank1/nitrate.parquet")
phosphate_df = convert_df("../../data/tank1/phosphate.parquet")
oxygen_df = convert_df("../../data/tank1/oxygen_a.parquet")
oxygen_df2 = convert_df("../../data/tank1/oxygen_b.parquet")
energy_df = convert_df("../../data/tank1/energy.parquet")
water_df = pd.read_csv("../../data/tank1/water.csv", delimiter=";")

# Preprocessing water data
water_df.index = pd.to_datetime(water_df['DateTime'], format='%d-%m-%Y %H:%M')
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].str.replace(',', '.').replace('(null)', np.nan, regex=True).astype(float)
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].interpolate()
water_df = water_df['EDE_09902MTW_K100.MTW']
water_df.index.name = None
water_df = water_df[water_df.index.isin(oxygen_df.index)]
water_df = water_df[~water_df.index.duplicated()]

# Synchronizing data based on datetime index
ammonium_df = ammonium_df[ammonium_df.index.isin(water_df.index)]
nitrate_df = nitrate_df[nitrate_df.index.isin(water_df.index)]
phosphate_df = phosphate_df[phosphate_df.index.isin(water_df.index)]
oxygen_df = oxygen_df[oxygen_df.index.isin(water_df.index)]
oxygen_df2 = oxygen_df2[oxygen_df2.index.isin(water_df.index)]
energy_df = energy_df[energy_df.index.isin(water_df.index)]

oxygen_df = (oxygen_df + oxygen_df2) / 2

# Merging all dataframes
merged_df = pd.concat([phosphate_df, 
                       ammonium_df, 
                       nitrate_df, 
                       energy_df,
                       water_df,
                       oxygen_df], axis=1)

merged_df.columns = ['Phosphate', 'Ammonium', 'Nitrate', 'Energy', 'WaterFlow', 'Oxygen']

# Feature Engineering: Time-related features
merged_df['HourOfDay'] = merged_df.index.hour
merged_df['Minute'] = merged_df.index.minute
merged_df['DayOfWeek'] = merged_df.index.dayofweek
merged_df['MonthOfYear'] = merged_df.index.month
merged_df['Season'] = (merged_df.index.month % 12 + 3) // 3

# Model preparation
X = merged_df.drop(['Oxygen'], axis=1)
y = merged_df['Oxygen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Use the best model to make predictions
params = {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
dt_model = DecisionTreeRegressor(**params, random_state=42)
dt_model.fit(X_train, y_train)

# Building and evaluating models
lr_model = LinearRegression().fit(X_train, y_train)

ridge_model = Ridge(alpha=100) 
ridge_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_ensemble = 0.23 * y_pred_lr + 0.75 * y_pred_dt + 0.02 * y_pred_ridge

# Model Evaluation
mse_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
print(f"Ensemble Model - MAE: {mse_ensemble}, R-squared: {r2_ensemble}")

# Saving results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ensemble}, index=y_test.index)
joblib.dump(results_df, "results_updated_whitebox.joblib")
joblib.dump(lr_model, "linear.joblib")
joblib.dump(ridge_model, "ridge.joblib")
joblib.dump(dt_model, "decision.joblib")
