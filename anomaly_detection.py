# anomaly_detection.py

"""
Anomaly Detection Module

This module contains functions to detect anomalies in the operational data.
It implements both statistical methods and machine learning algorithms for
anomaly detection in time-series data.

Functions:
- detect_anomalies_statistical(data, z_threshold)
- detect_anomalies_isolation_forest(data)
- detect_anomalies_autoencoder(data, epochs, batch_size)
- plot_anomalies(data, anomalies)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import regularizers

def detect_anomalies_statistical(data, z_threshold=3):
    """
    Detect anomalies using statistical methods (Z-score).

    Parameters:
    - data (DataFrame): DataFrame containing numerical features.
    - z_threshold (float): Z-score threshold for identifying anomalies.

    Returns:
    - anomalies (DataFrame): DataFrame indicating anomalies for each feature.
    """
    try:
        data_mean = data.mean()
        data_std = data.std()

        # Calculate Z-scores
        z_scores = (data - data_mean) / data_std

        # Identify anomalies
        anomalies = (np.abs(z_scores) > z_threshold)

        return anomalies
    except Exception as e:
        print(f"Error in statistical anomaly detection: {e}")
        return None

def detect_anomalies_isolation_forest(data, contamination=0.01):
    """
    Detect anomalies using Isolation Forest algorithm.

    Parameters:
    - data (DataFrame): DataFrame containing numerical features.
    - contamination (float): The proportion of outliers in the data set.

    Returns:
    - anomaly_scores (Series): Anomaly scores for each data point.
    - anomalies (Series): Boolean Series indicating anomalies.
    """
    try:
        # Initialize Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(data)

        # Compute anomaly scores
        anomaly_scores = iso_forest.decision_function(data)
        anomalies = iso_forest.predict(data)

        # Convert predictions to Boolean (1: normal, -1: anomaly)
        anomalies = pd.Series(anomalies, index=data.index)
        anomalies = anomalies.map({1: False, -1: True})

        return anomaly_scores, anomalies
    except Exception as e:
        print(f"Error in Isolation Forest anomaly detection: {e}")
        return None, None

def detect_anomalies_autoencoder(data, epochs=20, batch_size=64):
    """
    Detect anomalies using an Autoencoder neural network.

    Parameters:
    - data (DataFrame): DataFrame containing numerical features.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - reconstruction_errors (Series): Reconstruction error for each data point.
    - anomalies (Series): Boolean Series indicating anomalies.
    """
    try:
        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Split data into training and testing (use first 80% for training)
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]

        # Build Autoencoder model
        input_dim = train_data.shape[1]
        model = Sequential([
            Dense(16, activation='relu', input_shape=(input_dim,)),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(train_data, train_data,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(test_data, test_data),
                  shuffle=True,
                  verbose=1)

        # Compute reconstruction errors
        test_predictions = model.predict(test_data)
        mse = np.mean(np.power(test_data - test_predictions, 2), axis=1)
        reconstruction_errors = pd.Series(mse, index=data.index[train_size:])

        # Determine anomaly threshold (e.g., 95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)
        anomalies = reconstruction_errors > threshold

        return reconstruction_errors, anomalies
    except Exception as e:
        print(f"Error in Autoencoder anomaly detection: {e}")
        return None, None

def plot_anomalies(data, anomalies, feature):
    """
    Plot anomalies detected in a specific feature over time.

    Parameters:
    - data (DataFrame): Original data including timestamps.
    - anomalies (Series or DataFrame): Boolean Series indicating anomalies.
    - feature (str): The feature to plot.
    """
    try:
        plt.figure(figsize=(15, 5))
        plt.plot(data['timestamp'], data[feature], label='Normal')
        plt.scatter(
            data['timestamp'][anomalies],
            data[feature][anomalies],
            color='red',
            label='Anomaly'
        )
        plt.title(f'Anomalies in {feature} over Time')
        plt.xlabel('Timestamp')
        plt.ylabel(feature)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error in plotting anomalies: {e}")

def combine_anomaly_results(*anomaly_lists):
    """
    Combine multiple anomaly detection results into a single indicator.

    Parameters:
    - *anomaly_lists: Multiple Boolean Series or DataFrames indicating anomalies.

    Returns:
    - combined_anomalies (Series): Boolean Series indicating combined anomalies.
    """
    try:
        combined_anomalies = anomaly_lists[0]
        for anomalies in anomaly_lists[1:]:
            combined_anomalies = combined_anomalies | anomalies
        return combined_anomalies
    except Exception as e:
        print(f"Error in combining anomaly results: {e}")
        return None
