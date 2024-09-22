# lstm_model.py

"""
LSTM Model Module

This module contains functions to build, train, save, and load the LSTM model
used for predicting motor failures. It also includes functions for making
predictions with the trained model.

Functions:
- create_lstm_model(input_shape)
- train_lstm_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
- save_lstm_model(model, file_path)
- load_lstm_model(file_path)
- predict_with_lstm(model, X_test)
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_lstm_model(input_shape):
    """
    Build and compile the LSTM model.

    Parameters:
    - input_shape (tuple): Shape of the input data (sequence_length, num_features).

    Returns:
    - model (Sequential): Compiled LSTM model.
    """
    try:
        model = Sequential()

        # First LSTM layer with Dropout regularization
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))

        # Output layer with sigmoid activation for multi-label classification
        model.add(Dense(units=3, activation='sigmoid'))

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        print(f"Error in creating LSTM model: {e}")
        return None

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """
    Train the LSTM model with early stopping and model checkpointing.

    Parameters:
    - model (Sequential): The LSTM model to train.
    - X_train (ndarray): Training input sequences.
    - y_train (ndarray): Training target variables.
    - X_val (ndarray): Validation input sequences.
    - y_val (ndarray): Validation target variables.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.

    Returns:
    - model (Sequential): Trained LSTM model.
    - history (History): Training history object.
    """
    try:
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        return model, history
    except Exception as e:
        print(f"Error in training LSTM model: {e}")
        return None, None

def save_lstm_model(model, file_path):
    """
    Save the trained LSTM model to a file.

    Parameters:
    - model (Sequential): The trained LSTM model to save.
    - file_path (str): Path where the model will be saved.
    """
    try:
        model.save(file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error in saving LSTM model: {e}")

def load_lstm_model(file_path):
    """
    Load a saved LSTM model from a file.

    Parameters:
    - file_path (str): Path to the saved model file.

    Returns:
    - model (Sequential): Loaded LSTM model.
    """
    try:
        if os.path.exists(file_path):
            model = load_model(file_path)
            print(f"Model loaded from {file_path}")
            return model
        else:
            print(f"Model file not found at {file_path}")
            return None
    except Exception as e:
        print(f"Error in loading LSTM model: {e}")
        return None

def predict_with_lstm(model, X_test):
    """
    Make predictions using the trained LSTM model.

    Parameters:
    - model (Sequential): Trained LSTM model.
    - X_test (ndarray): Input sequences for prediction.

    Returns:
    - predictions (ndarray): Predicted probabilities.
    """
    try:
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"Error in making predictions with LSTM model: {e}")
        return None
