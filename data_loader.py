# data_loader.py

"""
Data Loader Module

This module contains functions to load various datasets required for the
motor failure prediction system. It handles the loading of operational data,
environmental data, failure records, maintenance logs, and operator notes.

Functions:
- load_csv_data(file_path, expected_columns=None, parse_dates=None)
- merge_data_on_timestamp(data_list)
- load_and_merge_data(file_paths)
"""

import pandas as pd
import logging

def load_csv_data(file_path, expected_columns=None, parse_dates=None):
    """
    Load data from a CSV file dynamically.

    Parameters:
    - file_path (str): Path to the CSV file.
    - expected_columns (list, optional): List of expected columns.
    - parse_dates (list, optional): List of columns to parse as dates.

    Returns:
    - data (DataFrame): Pandas DataFrame containing the data.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=parse_dates)
        data_columns = set(data.columns)

        if expected_columns:
            missing_cols = set(expected_columns) - data_columns
            if missing_cols:
                logging.warning(f"Missing columns in {file_path}: {missing_cols}")
            else:
                logging.info(f"All expected columns are present in {file_path}.")

        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def merge_data_on_timestamp(data_list, merge_type='inner', tolerance='1min'):
    """
    Merge multiple DataFrames on the 'timestamp' column.

    Parameters:
    - data_list (list): List of DataFrames to merge.
    - merge_type (str): Type of merge ('inner', 'outer', 'left', 'right').
    - tolerance (str): Tolerance for matching timestamps (e.g., '1min').

    Returns:
    - merged_data (DataFrame): Merged DataFrame.
    """
    try:
        from functools import reduce

        # Ensure all DataFrames are sorted by 'timestamp'
        for data in data_list:
            data.sort_values('timestamp', inplace=True)

        # Merge all DataFrames on 'timestamp' using nearest merge with tolerance
        merged_data = reduce(
            lambda left, right: pd.merge_asof(
                left, right, on='timestamp', direction='nearest', tolerance=pd.Timedelta(tolerance)
            ),
            data_list
        )

        return merged_data
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return None

def load_and_merge_data(file_paths):
    """
    Load and merge multiple datasets based on 'timestamp'.

    Parameters:
    - file_paths (dict): Dictionary with keys as data identifiers and values as file paths.

    Returns:
    - merged_data (DataFrame): Merged DataFrame ready for preprocessing.
    """
    data_list = []
    for key, file_info in file_paths.items():
        file_path = file_info['path']
        expected_columns = file_info.get('expected_columns', None)
        parse_dates = file_info.get('parse_dates', ['timestamp'])

        data = load_csv_data(file_path, expected_columns, parse_dates)
        if data is not None:
            data_list.append(data)
        else:
            logging.error(f"Failed to load data for {key} from {file_path}.")
            return None

    if data_list:
        merged_data = merge_data_on_timestamp(data_list)
        return merged_data
    else:
        logging.error("No data loaded to merge.")
        return None
