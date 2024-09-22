# utils.py

"""
Utilities Module

This module contains utility functions that are used across multiple modules
in the motor failure prediction system. Functions include saving and loading
scalers and models, handling configurations, and setting up logging.

Functions:
- save_scaler(scaler, file_path)
- load_scaler(file_path)
- save_model(model, file_path)
- load_model(file_path)
- load_config(config_file)
- setup_logging(log_file)
- create_directory_if_not_exists(directory)
"""

import os
import joblib
import json
import logging
from tensorflow.keras.models import load_model as keras_load_model

def save_scaler(scaler, file_path):
    """
    Save a scaler object to a file using joblib.
    
    Parameters:
    - scaler: The scaler object to save.
    - file_path (str): Path where the scaler will be saved.
    """
    try:
        joblib.dump(scaler, file_path)
        logging.info(f"Scaler saved to {file_path}")
    except Exception as e:
        logging.error(f"Error in saving scaler: {e}")

def load_scaler(file_path):
    """
    Load a scaler object from a file.
    
    Parameters:
    - file_path (str): Path to the saved scaler file.
    
    Returns:
    - scaler: The loaded scaler object.
    """
    try:
        scaler = joblib.load(file_path)
        logging.info(f"Scaler loaded from {file_path}")
        return scaler
    except Exception as e:
        logging.error(f"Error in loading scaler: {e}")
        return None

def save_model(model, file_path):
    """
    Save a Keras model to a file.
    
    Parameters:
    - model: The Keras model to save.
    - file_path (str): Path where the model will be saved.
    """
    try:
        model.save(file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error in saving model: {e}")

def load_model(file_path):
    """
    Load a Keras model from a file.
    
    Parameters:
    - file_path (str): Path to the saved model file.
    
    Returns:
    - model: The loaded Keras model.
    """
    try:
        if os.path.exists(file_path):
            model = keras_load_model(file_path)
            logging.info(f"Model loaded from {file_path}")
            return model
        else:
            logging.error(f"Model file not found at {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error in loading model: {e}")
        return None

def load_config(config_file):
    """
    Load configuration settings from a JSON file.
    
    Parameters:
    - config_file (str): Path to the configuration JSON file.
    
    Returns:
    - config (dict): Dictionary containing configuration settings.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logging.error(f"Error in loading configuration: {e}")
        return None

def setup_logging(log_file='app.log'):
    """
    Set up logging configuration.
    
    Parameters:
    - log_file (str): Path to the log file.
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging is set up.")
    except Exception as e:
        print(f"Error in setting up logging: {e}")

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not exist.
    
    Parameters:
    - directory (str): Path to the directory.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Directory created: {directory}")
        else:
            logging.info(f"Directory already exists: {directory}")
    except Exception as e:
        logging.error(f"Error in creating directory: {e}")
