# explainability.py

"""
Explainability Module

This module contains functions to implement explainable AI techniques for interpreting
the LSTM model's predictions. It uses SHAP values to compute feature importance and
integrates with the language model to simplify technical explanations.

Functions:
- explain_predictions(model, X_sample)
- simplify_explanation(language_model, tokenizer, technical_explanation)
- plot_feature_importance(shap_values, feature_names)
- generate_explainability_report(language_model, tokenizer, shap_values, X_sample)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Import necessary modules from other parts of the system
from llama_model import simplify_explanation

def explain_predictions(model, X_sample):
    """
    Compute SHAP values to explain the LSTM model's predictions.

    Parameters:
    - model: The trained LSTM model.
    - X_sample (ndarray): A sample of input sequences for which to compute explanations.

    Returns:
    - shap_values (list): A list of arrays containing SHAP values for each class.
    """
    try:
        # Initialize the DeepExplainer
        explainer = shap.DeepExplainer(model, X_sample)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    except Exception as e:
        print(f"Error in computing SHAP values: {e}")
        return None

def plot_feature_importance(shap_values, feature_names, class_index=0):
    """
    Plot the feature importance using SHAP values for a specific class.

    Parameters:
    - shap_values (list): A list of arrays containing SHAP values for each class.
    - feature_names (list): List of feature names corresponding to the input data.
    - class_index (int): Index of the class for which to plot feature importance.

    Returns:
    - None
    """
    try:
        # Aggregate SHAP values over the time steps
        shap_values_sum = np.sum(shap_values[class_index], axis=1)
        
        # Create a DataFrame for SHAP values
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': np.mean(np.abs(shap_values_sum), axis=0)
        })
        
        # Sort features by importance
        shap_df.sort_values(by='SHAP Value', ascending=False, inplace=True)
        
        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(shap_df['Feature'], shap_df['SHAP Value'])
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'Feature Importance for Class {class_index}')
        plt.gca().invert_yaxis()
        plt.show()
    except Exception as e:
        print(f"Error in plotting feature importance: {e}")

def generate_explainability_report(language_model, tokenizer, shap_values, X_sample, feature_names):
    """
    Generate a simplified explainability report using the language model.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - shap_values (list): A list of arrays containing SHAP values for each class.
    - X_sample (ndarray): The input sample data used for computing SHAP values.
    - feature_names (list): List of feature names corresponding to the input data.

    Returns:
    - report (str): A simplified explainability report.
    """
    try:
        # Generate technical explanation
        technical_explanation = create_technical_explanation(shap_values, X_sample, feature_names)
        
        # Simplify the explanation using the language model
        simplified_explanation = simplify_explanation(language_model, tokenizer, technical_explanation)
        
        return simplified_explanation
    except Exception as e:
        print(f"Error in generating explainability report: {e}")
        return "An error occurred while generating the explainability report."

def create_technical_explanation(shap_values, X_sample, feature_names):
    """
    Create a technical explanation based on SHAP values.

    Parameters:
    - shap_values (list): A list of arrays containing SHAP values for each class.
    - X_sample (ndarray): The input sample data used for computing SHAP values.
    - feature_names (list): List of feature names corresponding to the input data.

    Returns:
    - technical_explanation (str): A detailed technical explanation.
    """
    try:
        # Aggregate SHAP values over samples and time steps
        shap_values_mean = np.mean(np.abs(np.sum(shap_values[0], axis=1)), axis=0)
        
        # Create a DataFrame for SHAP values
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': shap_values_mean
        })
        
        # Sort features by importance
        shap_df.sort_values(by='Mean SHAP Value', ascending=False, inplace=True)
        
        # Generate a textual explanation
        explanation_lines = ["Technical Explanation of Model Predictions:"]
        for index, row in shap_df.iterrows():
            line = f"Feature '{row['Feature']}' has a mean SHAP value of {row['Mean SHAP Value']:.4f}, indicating its importance in the model's predictions."
            explanation_lines.append(line)
        
        technical_explanation = "\n".join(explanation_lines)
        return technical_explanation
    except Exception as e:
        print(f"Error in creating technical explanation: {e}")
        return "An error occurred while creating the technical explanation."

