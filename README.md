# Motor Failure Prediction System



## Overview

The Motor Failure Prediction System is an AI-driven application designed to predict motor     failures using a combination of Long Short-Term Memory (LSTM) neural networks and  language models. It leverages operational, environmental, and textual data (such as maintenance logs and operator notes) to forecast potential failures and provide insights into motor health.

  

## Key Features

**Data Preprocessing:** Cleans and prepares numerical and textual data for modeling.

**Feature Engineering:** Enhances the dataset with new features derived from existing data.

LSTM Model: Trains an LSTM neural network for time-series prediction of motor failures.

Language Model Integration: Uses a language model to interpret predictions and simplify technical explanations.

Explainability: Implements SHAP (SHapley Additive exPlanations) for model interpretability.

User Interface: Provides a command-line interface (CLI) for user interaction, including querying and report generation.

## Table of Contents

- Prerequisites
- Installation
- Project Structure
- Usage
	- Training Mode
	- Prediction Mode
	- Interactive Mode
- Configuration
- Adjustable Parameters
- Data Preparation
- Examples
- Troubleshooting
- Contributing
- License

## Prerequisites

**Python 3.7 or higher**

**pip** package installer

## Installation

### Clone the repository:

```bash
git clone ["link to my github repo"] cd motor-failure-prediction
```

### Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


### Install required packages:

```bash
pip install -r requirements.txt
```


## Project Structure

```markdown
motor-failure-prediction/
├── data/
│   ├── operational_data.csv
│   ├── environmental_data.csv
│   ├── failure_records.csv
│   ├── maintenance_logs.csv
│   └── operator_notes.csv
├── models/
│   ├── trained_lstm_model.h5
│   └── scaler.pkl
├── logs/
│   └── app.log
├── config/
│   └── config.json
├── main.py
├── data_loader.py
├── data_preprocessing.py
├── feature_engineering.py
├── lstm_model.py
├── llama_model.py
├── user_interface.py
├── explainability.py
├── utils.py
├── requirements.txt
└── README.md
```

## Usage

The application can be run in three modes:

### Training Mode

- Command: 
```bash
python main.py --mode train --config config/config.json
```
Description:
- Loads and preprocesses data.
- Performs feature engineering.
- Trains the LSTM model.
- Saves the trained model and scaler for future use.

### Prediction Mode

**Command:** 
```bash
python main.py --mode predict --config config/config.json
```
Description:
- Loads new operational data.
- Preprocesses and normalizes the data.
- Makes predictions using the trained model.
- Generates an explainability report.

### Interactive Mode

**Command:** 
```bash
python main.py --mode interact --config config/config.json
```
**Description:**
- Starts a CLI for user interaction.
- Allows querying motor status and generating reports.

## Configuration

The application uses a JSON configuration file located at config/config.json. This file contains settings for data paths, model parameters, and other configurations.

  

Example `config.json`
```json
{
    "batch_size": 64,
    "epochs": 50,
    "sequence_length": 50,
    "model_save_path": "models/trained_lstm_model.h5",
    "scaler_save_path": "models/scaler.pkl",
    "log_file": "logs/app.log",
    "data_paths": {
        "operational_data": "data/operational_data.csv",
        "environmental_data": "data/environmental_data.csv",
        "failure_records": "data/failure_records.csv",
        "maintenance_logs": "data/maintenance_logs.csv",
        "operator_notes": "data/operator_notes.csv"
    },
    "new_operational_data": "data/new_operational_data.csv",
    "new_environmental_data": "data/new_environmental_data.csv"
}

```

## Adjustable Parameters

The following parameters can be adjusted in the `config.json` file:

  

### Model Parameters

-   **`batch_size`**
    
    -   **Description**: Number of samples processed before the model is updated.
    -   **Default**: `64`
    -   **Effect**: Smaller batch sizes can lead to more stable training but longer training times.
-   **`epochs`**
    -   **Description**: Number of complete passes through the training dataset.
    -   **Default**: `50`
    -   **Effect**: More epochs can improve learning but may lead to overfitting.
-   **`sequence_length`**
    
    -   **Description**: Number of time steps used as input for the LSTM model.
    -   **Default**: `50`
    -   **Effect**: Longer sequences capture more temporal information but increase computational complexity.


### Data Paths

-   **`operational_data`**
    
    -   **Description**: Path to the operational data CSV file.
    -   **Default**: `"data/operational_data.csv"`
-   **`environmental_data`**
    
    -   **Description**: Path to the environmental data CSV file.
    -   **Default**: `"data/environmental_data.csv"`
-   **`failure_records`**
    
    -   **Description**: Path to the failure records CSV file.
    -   **Default**: `"data/failure_records.csv"`
-   **`maintenance_logs`**
    
    -   **Description**: Path to the maintenance logs CSV file.
    -   **Default**: `"data/maintenance_logs.csv"`
-   **`operator_notes`**
    
    -   **Description**: Path to the operator notes CSV file.
    -   **Default**: `"data/operator_notes.csv"`

### File Paths

-   **`model_save_path`**
    
    -   **Description**: Path where the trained model will be saved.
    -   **Default**: `"models/trained_lstm_model.h5"`
-   **`scaler_save_path`**
    
    -   **Description**: Path where the scaler will be saved.
    -   **Default**: `"models/scaler.pkl"`
-   **`log_file`**
    
    -   **Description**: Path to the application log file.
    -   **Default**: `"logs/app.log"`


## Data Preparation

Ensure that your data files are in the correct format and located at the paths specified in the configuration file.

### Required Data Files

1.  **Operational Data (`operational_data.csv`)**
 
    -   **Columns**: `timestamp`, `voltage`, `current`, `RPM`, etc.
3.  **Environmental Data (`environmental_data.csv`)**
   
    -   **Columns**: `timestamp`, `temperature`, `humidity`, etc.
4.  **Failure Records (`failure_records.csv`)**
    
    -   **Columns**: `timestamp`, `motor_id`, `failure_type`, etc.
5.  **Maintenance Logs (`maintenance_logs.csv`)**
    
    -   **Columns**: `timestamp`, `maintenance_action`, `technician_notes`
6.  **Operator Notes (`operator_notes.csv`)**
    
    -   **Columns**: `timestamp`, `note`

## Examples

### Training the Model

```bash
python main.py --mode train --config config/config.json
```
### Making Predictions

```bash
python main.py --mode predict --config config/config.json
```
### Interacting with the System

```bash
python main.py --mode interact --config config/config.json
```
  **Sample Interaction:**

```vbnet
Welcome to the Motor Health Monitoring System!
Type 'help' to see available commands. Type 'exit' to quit.

>> help

Available Commands:
help             - Show this help message.
exit             - Exit the system.
generate report  - Generate a comprehensive motor health report.
query <message>  - Ask a question about motor status or predictions.

Examples:
>> generate report
>> query What is the failure risk for Motor A in the next 7 days?

>> query What is the current health status of Motor B?

Response:
Motor B is currently operating within normal parameters with a low risk of failure in the next 7 days.

>> exit
Exiting the system. Goodbye!

```

## Troubleshooting

-   **Module Not Found Errors**
    
    -   Ensure all required packages are installed using `pip install -r requirements.txt`.
    -   Check your Python environment and ensure it's activated.
-   **Data File Errors**
    
    -   Verify that data files are in the correct directory and paths are correctly specified in `config.json`.
    -   Ensure data files are properly formatted CSVs with correct headers.
-   **Model Loading Errors**
    
    -   Ensure the model has been trained and the model files exist in the `models/` directory.
    -   Verify that the paths in `config.json` are correct.
-   **Memory Issues**
    
    -   For large datasets, consider increasing system memory or using a machine with more resources.
    -   Reduce batch size or sequence length to lower memory usage.

----------

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
I would appreciate any feedback and optimize my code!

----------