# feature_engineering.py

"""
Feature Engineering Module

This module contains functions to perform feature engineering on the preprocessed data.
It focuses on integrating unsupervised learning methods like clustering and incorporating
features suggested by the language model.

Functions:
- create_additional_features(data, language_model, tokenizer)
- perform_clustering(data, n_clusters)
- name_clusters(language_model, tokenizer, data, cluster_centers)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging


def create_additional_features(data, language_model, tokenizer):
    """
    Suggest additional features from existing data using the language model.

    Parameters:
    - data (DataFrame): The preprocessed data.
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.

    Returns:
    - data_with_features (DataFrame): DataFrame with new features added.
    """
    try:
        # Generate dataset summary dynamically from the data's columns
        feature_list = data.columns.tolist()
        dataset_summary = 'Features: ' + ', '.join(feature_list) + '.'

        # Prepare the prompt for the language model
        prompt = f"The dataset has the following {len(feature_list)} features: {', '.join(feature_list)}. Suggest useful feature engineering techniques to enhance predictive modeling, considering interactions, transformations, or aggregations."

        # Generate suggestions using the language model
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(language_model.device)
        attention_mask = inputs['attention_mask'].to(language_model.device)

        output_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        suggestions = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Log or print the suggestions for review
        print("Feature Engineering Suggestions from the Language Model:")
        print(suggestions)

        # Manually implement selected suggestions
        # For example, check for specific suggestions and implement accordingly

        # Convert suggestions to lowercase for consistent comparison
        suggestions_lower = suggestions.lower()

        # Implement interaction terms
        if "interaction terms" in suggestions_lower:
            # Example interaction between 'voltage' and 'current'
            if 'voltage' in data.columns and 'current' in data.columns:
                data['voltage_current_interaction'] = data['voltage'] * data['current']
                print("Added feature: voltage_current_interaction")

        # Implement rolling averages
        if "rolling average" in suggestions_lower or "rolling mean" in suggestions_lower:
            # Example rolling mean of 'temperature'
            if 'temperature' in data.columns:
                data['temperature_rolling_mean_5'] = data['temperature'].rolling(window=5).mean()
                data['temperature_rolling_mean_5'].fillna(method='bfill', inplace=True)
                print("Added feature: temperature_rolling_mean_5")

        # Implement feature scaling if suggested
        if "standardize features" in suggestions_lower or "scale features" in suggestions_lower:
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            print("Standardized numerical features.")

        # Implement logarithmic transformations if suggested
        if "logarithmic transformation" in suggestions_lower:
            # Example: Apply log transformation to 'vibration' feature
            if 'vibration' in data.columns:
                data['log_vibration'] = np.log1p(data['vibration'])
                print("Added feature: log_vibration")

        # Add more conditions based on the suggestions and available features

        # Return the data with new features added
        return data
    except Exception as e:
        logging.error(f"Error in create_additional_features: {e}")
        return data

def perform_clustering(data, n_clusters):
    """
    Perform clustering on the data and add cluster labels as a feature.

    Parameters:
    - data (DataFrame): The data to cluster.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - data_with_clusters (DataFrame): DataFrame with cluster labels added as a feature.
    - cluster_centers (ndarray): Coordinates of cluster centers.
    """
    try:
        # Select numerical features for clustering
        features = data.select_dtypes(include=[np.number]).columns.tolist()

        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[features])

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)

        # Add cluster labels to data
        data['cluster_label'] = cluster_labels

        return data, kmeans.cluster_centers_
    except Exception as e:
        logging.error(f"Error in perform_clustering: {e}")
        return data, None

def name_clusters(language_model, tokenizer, data, cluster_centers):
    """
    Use the language model to name clusters based on cluster centers.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - data (DataFrame): The data with cluster labels.
    - cluster_centers (ndarray): Coordinates of cluster centers.

    Returns:
    - data_with_cluster_names (DataFrame): DataFrame with cluster names added.
    - cluster_names (dict): Dictionary mapping cluster labels to names.
    """
    try:
        cluster_names = {}
        features = data.select_dtypes(include=[np.number]).columns.tolist()
        for idx, center in enumerate(cluster_centers):
            center_dict = dict(zip(features, center))
            description = generate_cluster_description(center_dict)
            prompt = f"Provide a concise name for a cluster with the following characteristics:\n{description}"
            inputs = tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            device = language_model.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate output
            output_ids = language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

            cluster_name = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            cluster_names[idx] = cluster_name.strip()

        # Map cluster names to data
        data['cluster_name'] = data['cluster_label'].map(cluster_names)

        return data, cluster_names
    except Exception as e:
        logging.error(f"Error in name_clusters: {e}")
        return data, {}

def generate_cluster_description(center_dict):
    """
    Generate a textual description of a cluster center.

    Parameters:
    - center_dict (dict): Dictionary of feature values for the cluster center.

    Returns:
    - description (str): Textual description of the cluster characteristics.
    """
    description = ""
    for feature, value in center_dict.items():
        description += f"{feature}: {value:.2f}\n"
    return description
