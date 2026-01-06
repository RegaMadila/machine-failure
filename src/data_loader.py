import pandas as pd
import os

import opendatasets as od

def load_data(filepath=None):
    dataset_url = "https://www.kaggle.com/stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    data_dir = "predictive-maintenance-dataset-ai4i-2020"
    csv_file = "ai4i2020.csv"
    
    full_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(full_path):
        print(f"Downloading dataset from {dataset_url}...")
        od.download(dataset_url)
    
    if not os.path.exists(full_path):
         raise FileNotFoundError(f"Failed to find {csv_file} in {data_dir} after download.")

    df = pd.read_csv(full_path)
    print(f"Data loaded successfully from {full_path}. Shape: {df.shape}")
    return df

def clean_data(df):

    df = df.rename(columns={
        'Air temperature [K]': 'air_temperature_k',
        'Process temperature [K]': 'process_temperature_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'Product ID': 'product_id',
        'Type': 'type'
    })
    
    cols_to_drop = ['UDI', 'product_id']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    
    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"Dropped columns: {existing_drop}")
        
    return df
