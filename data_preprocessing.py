# data_preprocessing.py

"""
Data Preprocessing Module

This module contains functions to preprocess the data for modeling.
It includes functions for handling missing values, encoding categorical variables,
labeling the data, normalizing features, and preparing data for LSTM models.

Functions:
- preprocess_numerical_data(data)
- label_data(data, failure_records)
- normalize_data(data, scaler=None)
- prepare_lstm_data(data, sequence_length)
- prepare_sequences(data, sequence_length)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

def preprocess_numerical_data(data):
    """
    Preprocess numerical data by handling missing values and removing outliers.

    Parameters:
    - data (DataFrame): The merged data.

    Returns:
    - preprocessed_data (DataFrame): Preprocessed data.
    """
    try:
        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Remove outliers using quantile-based flooring and capping
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_columns:
            lower_quantile = data[col].quantile(0.01)
            upper_quantile = data[col].quantile(0.99)
            data[col] = data[col].clip(lower=lower_quantile, upper=upper_quantile)

        return data
    except Exception as e:
        logging.error(f"Error in preprocess_numerical_data: {e}")
        return data

def label_data(data, failure_records, window=7):
    """
    Label the data with a target variable indicating failure within a certain time window.

    Parameters:
    - data (DataFrame): The data with features.
    - failure_records (DataFrame): DataFrame containing failure timestamps.
    - window (int): Time window in days to label failures.

    Returns:
    - labeled_data (DataFrame): Data with target variable added.
    """
    try:
        data['failure'] = 0
        failure_timestamps = failure_records['timestamp'].tolist()
        data_timestamps = data['timestamp']

        # Convert timestamps to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data_timestamps):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data_timestamps = data['timestamp']

        for failure_time in failure_timestamps:
            window_start = failure_time - pd.Timedelta(days=window)
            data.loc[(data_timestamps >= window_start) & (data_timestamps <= failure_time), 'failure'] = 1

        return data
    except Exception as e:
        logging.error(f"Error in label_data: {e}")
        return data

def normalize_data(data, scaler=None):
    """
    Normalize numerical features in the data using MinMaxScaler.

    Parameters:
    - data (DataFrame): The data to normalize.
    - scaler: A fitted scaler object. If None, a new scaler is created and fitted.

    Returns:
    - normalized_data (DataFrame): Data with numerical features normalized.
    - scaler: The scaler used for normalization.
    """
    try:
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'failure' in numerical_columns:
            numerical_columns.remove('failure')  # Exclude target variable

        if scaler is None:
            scaler = MinMaxScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        else:
            data[numerical_columns] = scaler.transform(data[numerical_columns])

        return data, scaler
    except Exception as e:
        logging.error(f"Error in normalize_data: {e}")
        return data, scaler

def prepare_lstm_data(data, sequence_length=50):
    """
    Prepare data for LSTM model by creating sequences.

    Parameters:
    - data (DataFrame): The data to prepare.
    - sequence_length (int): The length of the sequences.

    Returns:
    - X_train (ndarray): Training input sequences.
    - X_val (ndarray): Validation input sequences.
    - y_train (ndarray): Training target values.
    - y_val (ndarray): Validation target values.
    """
    try:
        from sklearn.model_selection import train_test_split

        # Remove any rows with missing values
        data = data.dropna()

        # Extract features and target
        features = data.drop(['timestamp', 'failure'], axis=1).values
        target = data['failure'].values

        # Create sequences
        X = []
        y = []
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(target[i])

        X = np.array(X)
        y = np.array(y)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        return X_train, X_val, y_train, y_val
    except Exception as e:
        logging.error(f"Error in prepare_lstm_data: {e}")
        return None, None, None, None

def prepare_sequences(data, sequence_length=50):
    """
    Prepare input sequences for prediction.

    Parameters:
    - data (DataFrame): The data to prepare.
    - sequence_length (int): The length of the sequences.

    Returns:
    - X (ndarray): Input sequences.
    """
    try:
        data = data.dropna()
        features = data.drop(['timestamp'], axis=1).values

        X = []
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])

        X = np.array(X)
        return X
    except Exception as e:
        logging.error(f"Error in prepare_sequences: {e}")
        return None
