import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

# Load and preprocess the data
def convert_df(path):
    dframe = pd.read_parquet(path)
    dframe.index = pd.to_datetime(dframe['datumBeginMeting'])
    dframe.index.name = None
    dframe.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
    dframe = dframe["hstWaarde"].astype(float)
    return dframe

def run():

    ammonium = convert_df("../data/tank1_new/ammonium.parquet")
    nitrate = convert_df("../data/tank1_new/nitrate.parquet")
    phosphate = convert_df("../data/tank1_new/phosphate.parquet")
    oxygen = convert_df("../data/tank1_new/oxygen_a.parquet")

    # Create a binary feature for oxygen
    oxygen_binary = oxygen >= 1

    # Merge all dataframes
    merged_df = pd.concat([oxygen, ammonium, nitrate, phosphate, oxygen_binary], axis=1, keys=['oxygen', 'ammonium', 'nitrate', 'phosphate', 'oxygen_binary'])

    # Prepare features and target
    X = merged_df[['ammonium', 'nitrate', 'phosphate', 'oxygen_binary']].values
    y = merged_df['oxygen'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False, stratify = None)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a Keras model
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(32, activation='relu'), # old 64
        layers.Dense(16, activation='relu'), # old 32
        layers.Dense(1)
    ])

    # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error') OLS
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    # model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2) OLD
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

    # Evaluate the model on the testing data
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")

    # Make predictions
    predictions = model.predict(X_test)
    print(predictions)

    joblib.dump(predictions, 'predictions.joblib')

def test():
    return joblib.load("predictions.joblib")

if __name__ == "__main__":
    run()
    np.savetxt('test.txt', test(), delimiter=',')