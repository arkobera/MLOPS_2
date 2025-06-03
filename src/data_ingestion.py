import pandas as pd
import numpy as np
import logging 
import os
import yaml

# Create logs directory if it doesn't exist
log_dir = "C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.DEBUG)
logger.addHandler(terminal_handler)

logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
terminal_handler.setFormatter(logger_formatter)

# Save log file in the logs directory
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)
# src/data_ingestion.py

def load_parameters(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully")
        print("Parameters loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        print(f"Error loading parameters: {e}")
        return {}


def load_data(train_path: str, test_path: str):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("Data loaded successfully")
        print("Data loaded successfully")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data.drop(columns=['id'], errors='ignore')
        data['Sex'] = data['Sex'].map({'female':0,"male":1})
        logger.info("Data preprocessed successfully")
        print("Data preprocessed successfully")
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        print(f"Error preprocessing data: {e}")
        return pd.DataFrame()

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    try:
        os.makedirs(output_path, exist_ok=True)
        train_data.to_csv(os.path.join(output_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(output_path, "test.csv"), index=False)
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

if __name__ == "__main__":
    params = load_parameters("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/params.yaml")
    train_path = params['data_ingestion']['train_file']
    test_path = params["data_ingestion"]["test_file"]
    output_path = params["data_ingestion"]["output_dir"]

    try:
        train_data, test_data = load_data(train_path, test_path)
        if not train_data.empty and not test_data.empty:
            train_data = preprocess_data(train_data)
            test_data = preprocess_data(test_data)
            save_data(train_data, test_data, output_path)
            print("Data saved successfully")
        else:
            logger.warning("No data to process")
            print("No data to process")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        print(f"An error occurred: {e}")
        raise