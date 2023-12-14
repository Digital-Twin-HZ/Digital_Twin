import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

results_df = joblib.load("results.joblib").sort_index()

# Filtering the DataFrame to the last four months
end_date = results_df.index.max()
start_date = end_date - pd.DateOffset(days=5)
results_df_last_4_months = results_df.loc[start_date:end_date]

# Plotting the time series for the last four months
plt.figure(figsize=(10, 6))
plt.plot(results_df_last_4_months['Actual'], label='Actual Oxygen Levels', alpha=0.7)
plt.plot(results_df_last_4_months['Predicted'], label='Predicted Oxygen Levels', alpha=0.7)
plt.xlabel('Timestamp')
plt.ylabel('Oxygen Levels')
plt.title('Time Series of Actual vs Predicted Oxygen Levels (Last 8 Hours)')
plt.legend()
plt.show()

print(mean_absolute_error(results_df['Actual'], results_df['Predicted']))