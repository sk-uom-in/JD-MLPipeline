
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined training and testing script for predicting machine status ahead of time.
This script performs:
  1. Data Preprocessing:
       - Reads sensor data from a CSV.
       - Drops columns with >70% missing values.
       - Parses and sorts by timestamp.
       - Interpolates and fills missing numeric values.
       - **Drops timestamp/date columns.**
       - **Caps outliers using the Median Absolute Deviation (MAD) method.**
       - **Creates derivative features (first differences) for each sensor reading.**
       - Scales the sensor features.
       - Creates sliding windows (with a configurable lead time for prediction).
  2. Model Definition:
       - A stacked LSTM with an attention mechanism.
       - A deeper MLP block after attention (two hidden layers) for classification.
  3. Training:
       - Uses Adam with learning rate scheduling.
       - Saves the final model.
  4. Testing & Evaluation:
       - Evaluates test accuracy.
       - Plots both a frequency distribution (bar chart) and a time-series plot (for a subset) comparing actual vs. predicted classes.
       
Adjust parameters as needed.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


############################################
# 1. Data Preprocessing Function with Enhancements
############################################

def cap_outliers(series, threshold=3.0):
    """
    Caps outliers in a pandas Series using the Median Absolute Deviation (MAD) method.
    Values beyond [median - threshold*MAD, median + threshold*MAD] are capped.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    # Avoid division by zero in case mad is zero.
    if mad == 0:
        return series
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    return series.clip(lower=lower_bound, upper=upper_bound)


def preprocess_data(file_path, window_size, lead_time, test_split):
    """
    Preprocess the sensor data.
    
    Parameters:
        file_path (str): Path to the CSV file.
        window_size (int): Number of consecutive rows (minutes) for each sliding window.
        lead_time (int): How many minutes ahead to predict the machine status.
        test_split (float): Fraction of sliding windows reserved for testing.
    
    Returns:
        X_train (torch.Tensor): Training inputs of shape (num_train_windows, window_size, num_features)
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Test inputs.
        y_test (torch.Tensor): Test labels.
        scaler (StandardScaler): Fitted scaler.
        label_encoder (LabelEncoder): Fitted label encoder.
        df_processed (pd.DataFrame): The processed DataFrame.
    """
    # Read CSV and parse timestamp.
    df = pd.read_csv(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    else:
        print("Warning: No timestamp column found; proceeding without time-sorting.")
    df.reset_index(drop=True, inplace=True)
    
    # Drop columns with >70% missing values.
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > 0.7]
    if cols_to_drop:
        print(f"Dropping columns due to >70% missing: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    
    # Set timestamp as index for time-based interpolation if present.
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Identify numeric columns (exclude machine_status if present).
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Interpolate numeric columns.
    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both')
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Reset index.
    df.reset_index(inplace=True)
    
    # Filter valid machine_status rows.
    valid_status = ['NORMAL', 'BROKEN', 'RECOVERING']
    if 'machine_status' in df.columns:
        initial_len = len(df)
        df = df[df['machine_status'].isin(valid_status)]
        final_len = len(df)
        if final_len < initial_len:
            print(f"Filtered out {initial_len - final_len} rows with invalid machine_status values.")
    else:
        raise ValueError("Column 'machine_status' not found in data.")
    
    # --- Outlier Capping using MAD ---
    # Identify candidate sensor columns: all numeric columns except machine_status.
    sensor_candidates = [col for col in df.columns if col not in ['machine_status'] and
                         not (('time' in col.lower()) or ('date' in col.lower()))]
    for col in sensor_candidates:
        df[col] = cap_outliers(df[col], threshold=3.0)
    
    # --- Feature Engineering: Derivative Features ---
    # For each sensor candidate, create a derivative (first difference) feature.
    for col in sensor_candidates:
        df[col + '_diff'] = df[col].diff().fillna(0)
    
    # --- Drop timestamp/date columns (if present) ---
    drop_time_cols = [col for col in df.columns if ('time' in col.lower()) or ('date' in col.lower())]
    if drop_time_cols:
        df.drop(columns=drop_time_cols, inplace=True)
    
    # Update sensor columns: everything except machine_status.
    sensor_cols = [col for col in df.columns if col != 'machine_status']
    
    # --- Scaling ---
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    
    # --- Encode machine_status ---
    label_encoder = LabelEncoder()
    df['machine_status'] = label_encoder.fit_transform(df['machine_status'])
    
    # --- Create Sliding Windows ---
    sensor_data = df[sensor_cols].values  # shape: (num_rows, num_features)
    labels = df['machine_status'].values    # shape: (num_rows,)
    num_rows = sensor_data.shape[0]
    
    X_windows = []
    y_windows = []
    max_start = num_rows - window_size - lead_time + 1
    for i in range(max_start):
        window = sensor_data[i : i + window_size]
        label_index = i + window_size - 1 + lead_time
        if label_index < num_rows:
            X_windows.append(window)
            y_windows.append(labels[label_index])
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    print(f"Created {X_windows.shape[0]} sliding windows, each of shape {X_windows.shape[1:]}.")
    
    # --- Train/Test Split (time-series split) ---
    split_index = int((1 - test_split) * X_windows.shape[0])
    X_train = X_windows[:split_index]
    y_train = y_windows[:split_index]
    X_test = X_windows[split_index:]
    y_test = y_windows[split_index:]
    print(f"Train windows: {X_train.shape[0]}, Test windows: {X_test.shape[0]}")
    
    # Convert to PyTorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test, scaler, label_encoder, df