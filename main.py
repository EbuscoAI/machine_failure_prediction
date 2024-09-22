# main.py

"""
Main Script

This script serves as the entry point for the motor failure prediction system.
It orchestrates the workflow by integrating all modules, handling configurations,
and providing command-line interface options for different operations.

Usage:
    python main.py --mode train
    python main.py --mode predict
    python main.py --mode interact

Command-Line Arguments:
    --mode          Operation mode: 'train', 'predict', 'interact'
    --config        Path to the configuration file (default: 'config/config.json')
"""

import sys
import argparse
import logging
import pandas as pd

# Import custom modules
from data_loader import load_csv_data, merge_data_on_timestamp
from data_preprocessing import preprocess_numerical_data, label_data, normalize_data, prepare_lstm_data, prepare_sequences
from feature_engineering import create_additional_features, incorporate_llama_features, perform_clustering, name_clusters
from lstm_model import create_lstm_model, train_lstm_model, save_lstm_model, load_lstm_model, predict_with_lstm
from llama_model import initialize_language_model, fine_tune_language_model
from user_interface import start_cli
from explainability import explain_predictions, plot_feature_importance, generate_explainability_report
from utils import save_scaler, load_scaler, load_config, setup_logging, create_directory_if_not_exists

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Motor Failure Prediction System')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'interact'],
                        help='Operation mode: train, predict, interact')
    parser.add_argument('--config', type=str, default='config/config.json',
                        help='Path to the configuration file')
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)
    if config is None:
        logging.error("Configuration could not be loaded. Exiting.")
        sys.exit(1)

    device = config.get('device', 'cpu')
    global DEVICE
    DEVICE = device

    # Ensure necessary directories exist
    create_directory_if_not_exists('models')
    create_directory_if_not_exists('logs')
    create_directory_if_not_exists('data')

    # Handle different modes
    if args.mode == 'train':
        train_mode(config, device)
    elif args.mode == 'predict':
        predict_mode(config, device)
    elif args.mode == 'interact':
        interact_mode(config, device)
    else:
        logging.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

def train_mode(config, device):
    """
    Train the LSTM model using the provided data and configuration.
    """
    logging.info("Starting training mode.")

    # Load data paths from configuration
    data_paths = config.get('data_paths', {})

    # Prepare file_paths dictionary for load_and_merge_data
    file_paths = {
        'operational_data': {
            'path': data_paths.get('operational_data'),
            'parse_dates': ['timestamp']
        },
        'environmental_data': {
            'path': data_paths.get('environmental_data'),
            'parse_dates': ['timestamp']
        },
        'failure_records': {
            'path': data_paths.get('failure_records'),
            'parse_dates': ['timestamp']
        },
        'maintenance_logs': {
            'path': data_paths.get('maintenance_logs'),
            'parse_dates': ['timestamp']
        },
        'operator_notes': {
            'path': data_paths.get('operator_notes'),
            'parse_dates': ['timestamp']
        }
    }

    # Load and merge operational and environmental data
    data_list = []
    for key in ['operational_data', 'environmental_data']:
        data_info = file_paths[key]
        data = load_csv_data(
            data_info['path'],
            parse_dates=data_info.get('parse_dates')
        )
        if data is not None:
            data_list.append(data)
        else:
            logging.error(f"Failed to load {key}")
            sys.exit(1)

    merged_data = merge_data_on_timestamp(data_list)

    if merged_data is None:
        logging.error("Failed to merge operational and environmental data.")
        sys.exit(1)

    # Load failure records, maintenance logs, operator notes
    failure_records = load_csv_data(
        file_paths['failure_records']['path'],
        parse_dates=file_paths['failure_records'].get('parse_dates')
    )
    maintenance_logs = load_csv_data(
        file_paths['maintenance_logs']['path'],
        parse_dates=file_paths['maintenance_logs'].get('parse_dates')
    )
    operator_notes = load_csv_data(
        file_paths['operator_notes']['path'],
        parse_dates=file_paths['operator_notes'].get('parse_dates')
    )

    # Preprocess numerical data
    preprocessed_data = preprocess_numerical_data(merged_data)

    # Initialize and fine-tune the language model
    language_model, tokenizer = initialize_language_model(device=device)
    # Combine maintenance logs and operator notes
    textual_data = pd.concat([maintenance_logs, operator_notes], ignore_index=True)
    domain_data = textual_data['note'].tolist()  # Adjust based on your data
    language_model = fine_tune_language_model(language_model, tokenizer, domain_data, device=device)

    # Perform feature engineering
    data_with_features = create_additional_features(preprocessed_data, language_model, tokenizer)

    # Perform clustering and name clusters
    n_clusters = config.get('n_clusters', 5)
    data_with_clusters, cluster_centers = perform_clustering(data_with_features, n_clusters=n_clusters)
    data_with_named_clusters, cluster_names = name_clusters(language_model, tokenizer, data_with_clusters, cluster_centers)

    # Incorporate features suggested by the language model
    data_with_all_features = incorporate_llama_features(data_with_named_clusters, language_model, tokenizer, textual_data)

    # Label the data with target variables
    labeled_data = label_data(data_with_all_features, failure_records)

    # Normalize the data
    normalized_data, scaler = normalize_data(labeled_data)
    save_scaler(scaler, config.get('scaler_save_path', 'models/scaler.pkl'))

    # Prepare data for LSTM
    sequence_length = config.get('sequence_length', 50)
    X_train, X_val, y_train, y_val = prepare_lstm_data(normalized_data, sequence_length=sequence_length)

    if X_train is None:
        logging.error("Failed to prepare LSTM data.")
        sys.exit(1)

    # Create the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
    model = create_lstm_model(input_shape)

    # Train the model
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 64)
    model, history = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    model_save_path = config.get('model_save_path', 'models/trained_lstm_model.h5')
    save_lstm_model(model, model_save_path)

    logging.info("Training completed successfully.")

def predict_mode(config, device):
    """
    Use the trained model to make predictions on new data.
    """
    logging.info("Starting prediction mode.")

    # Load new data paths from configuration
    new_operational_file = config.get('new_operational_data')
    new_environmental_file = config.get('new_environmental_data')

    # Prepare file_paths for load_and_merge_data
    file_paths = {
        'operational_data': {
            'path': new_operational_file,
            'parse_dates': ['timestamp']
        },
        'environmental_data': {
            'path': new_environmental_file,
            'parse_dates': ['timestamp']
        }
    }

    # Load and merge new operational and environmental data
    data_list = []
    for key in ['operational_data', 'environmental_data']:
        data_info = file_paths[key]
        data = load_csv_data(
            data_info['path'],
            parse_dates=data_info.get('parse_dates')
        )
        if data is not None:
            data_list.append(data)
        else:
            logging.error(f"Failed to load {key}")
            sys.exit(1)

    merged_data = merge_data_on_timestamp(data_list)

    if merged_data is None:
        logging.error("Failed to merge new operational and environmental data.")
        sys.exit(1)

    # Preprocess numerical data
    preprocessed_data = preprocess_numerical_data(merged_data)

    # Initialize the language model
    language_model, tokenizer = initialize_language_model(device=device)

    # Perform feature engineering
    data_with_features = create_additional_features(preprocessed_data, language_model, tokenizer)

    # Perform clustering and name clusters
    n_clusters = config.get('n_clusters', 5)
    data_with_clusters, cluster_centers = perform_clustering(data_with_features, n_clusters=n_clusters)
    data_with_named_clusters, cluster_names = name_clusters(language_model, tokenizer, data_with_clusters, cluster_centers)

    # Incorporate features suggested by the language model
    data_with_all_features = data_with_named_clusters  # Assuming no textual data for new data

    # Load the saved scaler and normalize data
    scaler_path = config.get('scaler_save_path', 'models/scaler.pkl')
    scaler = load_scaler(scaler_path)
    normalized_data, _ = normalize_data(data_with_all_features, scaler)

    # Prepare data for LSTM
    sequence_length = config.get('sequence_length', 50)
    X_new = prepare_sequences(normalized_data, sequence_length=sequence_length)

    if X_new is None:
        logging.error("Failed to prepare sequences for prediction.")
        sys.exit(1)

    # Load the trained model
    model_path = config.get('model_save_path', 'models/trained_lstm_model.h5')
    model = load_lstm_model(model_path)

    # Make predictions
    predictions = predict_with_lstm(model, X_new)

    # Generate explainability report
    shap_values = explain_predictions(model, X_new)
    feature_names = normalized_data.drop(['timestamp'], axis=1).columns.tolist()
    report = generate_explainability_report(language_model, tokenizer, shap_values, X_new, feature_names)

    # Display predictions and report
    print("Predictions:")
    print(predictions)
    print("\nExplainability Report:")
    print(report)

    # Plot feature importance
    plot_feature_importance(shap_values, feature_names, class_index=0)

    logging.info("Prediction completed successfully.")

def interact_mode(config, device):
    """
    Start the user interface for interactive queries and reports.
    """
    logging.info("Starting interactive mode.")

    # Initialize the language model
    language_model, tokenizer = initialize_language_model(device=device)

    # Load context data if needed
    context_data = {}  # Prepare any necessary context data

    # Start the command-line interface
    start_cli(language_model, tokenizer, context_data, device)

    logging.info("Interactive mode ended.")

if __name__ == "__main__":
    main()
