#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_from_json.py

This script demonstrates how to:
  1. Accept a JSON input (representing sensor readings along with metadata).
  2. Preprocess the data in a way analogous to the CSV pipeline (e.g., parsing timestamp,
     capping outliers, creating derivative features, scaling).
  3. Create a sliding window input (padding/replicating rows if needed) so that the
     input shape matches what the LSTM-Attention model was trained on.
  4. Load the saved model.
  5. Run inference and output the prediction in JSON format.
  
The model is assumed to have been trained using:
    - window_size = 10
    - hidden_size = 128
    - num_layers = 3
    - dropout = 0.3
    - num_classes = 3   (with the mapping: 0 -> "BROKEN", 1 -> "NORMAL", 2 -> "RECOVERING")
  
Make sure that the saved model file (here "final_model2.pt") is in the same directory.
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

############################################
# Helper Function: Outlier Capping using MAD
############################################
def cap_outliers(series, threshold=3.0):
    """
    Caps outliers in a pandas Series using the Median Absolute Deviation (MAD) method.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    # Avoid division by zero
    if mad == 0:
        return series
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    return series.clip(lower=lower_bound, upper=upper_bound)

############################################
# Model Definition: LSTM with Attention and MLP Block
############################################
class LSTMAttentionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_mlp=True):
        """
        LSTM-Attention network for time-series classification.
        """
        super(LSTMAttentionPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_mlp = use_mlp
        
        # Stacked LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism: a learnable attention vector.
        self.attention_vector = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        
        # Deeper MLP block for classification.
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, num_classes)
            )
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))  # shape: (batch_size, seq_length, hidden_size)
        energy = torch.tanh(lstm_out)          # (batch_size, seq_length, hidden_size)
        attn_scores = torch.matmul(energy, self.attention_vector)  # (batch_size, seq_length)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, seq_length, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_size)
        context = self.dropout(context)
        
        if self.use_mlp:
            output = self.mlp(context)
        else:
            output = self.fc(context)
        return output

############################################
# Preprocessing Function for JSON Input
############################################
def preprocess_json_data(json_data, window_size=10):
    """
    Preprocesses a JSON input into a sliding window suitable for model prediction.
    Steps:
      - Convert the JSON (assumed to be a dict) into a DataFrame.
      - Parse and sort the timestamp.
      - Cap outliers on sensor columns.
      - Create derivative (first-difference) features.
      - Drop timestamp/date columns.
      - Scale sensor features.
      - If there are not enough rows to form a window, replicate the record until
        a window of the desired size is obtained.
    
    Returns:
        X_window: A numpy array of shape (1, window_size, num_features)
                   ready for model inference.
        features: List of feature column names used.
    """
    # Convert the JSON record to a DataFrame (one row)
    df = pd.DataFrame([json_data])
    
    # Parse the timestamp if present and sort (for consistency)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    
    # Identify sensor columns (here: those that start with 'sensor_')
    sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
    
    # Apply outlier capping to sensor columns
    for col in sensor_columns:
        df[col] = cap_outliers(df[col])
    
    # Feature Engineering: Create derivative (first difference) features for each sensor column.
    for col in sensor_columns:
        df[col + '_diff'] = df[col].diff().fillna(0)
    
    # Drop timestamp/date columns (we no longer need them for model inference)
    drop_cols = [col for col in df.columns if ('time' in col.lower()) or ('date' in col.lower())]
    df.drop(columns=drop_cols, inplace=True)
    
    # Determine the feature set: all columns except 'machine_status' and 'device_id'
    features = [col for col in df.columns if col not in ['machine_status', 'device_id']]
    
    # Scaling: Fit a StandardScaler on these features.
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Create a sliding window:
    # If we have fewer rows than the required window_size, replicate the last row.
    num_rows = df.shape[0]
    if num_rows < window_size:
        # Replicate the last row until we have window_size rows.
        extra = pd.DataFrame([df.iloc[-1]] * (window_size - num_rows), columns=df.columns)
        df = pd.concat([df, extra], ignore_index=True)
    else:
        # If more than window_size rows exist, use the last window_size rows.
        df = df.tail(window_size).reset_index(drop=True)
    
    # Convert the feature values to a numpy array with shape (window_size, num_features)
    sensor_data = df[features].values
    # Reshape to (1, window_size, num_features) as expected by the model.
    X_window = sensor_data.reshape(1, window_size, -1)
    return X_window, features

############################################
# Main: Load JSON, Preprocess, Load Model, Predict, Output JSON
############################################
def main():
    # --- Example JSON input (as provided) ---
    json_input = {
        "device_id": "1",
        "timestamp": "2025-02-09T00:12:41.663204",
        "sensor_0": 2.465394,
        "sensor_1": 47.092009999999995,
        "sensor_2": 53.2118,
        "sensor_3": 46.310759999999995,
        "sensor_4": 634.375,
        "sensor_5": 76.45975,
        "sensor_6": 13.41146,
        "sensor_7": 16.13136,
        "sensor_8": 15.56713,
        "sensor_9": 15.05353,
        "sensor_10": 37.2274,
        "sensor_11": 47.52422,
        "sensor_12": 31.11716,
        "sensor_13": 1.681353,
        "sensor_14": 419.5747,
        "sensor_15": 0.0,
        "sensor_16": 461.8781,
        "sensor_17": 466.3284,
        "sensor_18": 2.565284,
        "sensor_19": 665.3993,
        "sensor_20": 398.9862,
        "sensor_21": 880.0001,
        "sensor_22": 498.8926,
        "sensor_23": 975.9409,
        "sensor_24": 627.674,
        "sensor_25": 741.7151,
        "sensor_26": 848.0708,
        "sensor_27": 429.0377,
        "sensor_28": 785.1935,
        "sensor_29": 684.9443,
        "sensor_30": 594.4445,
        "sensor_31": 682.8125,
        "sensor_32": 680.4416,
        "sensor_33": 433.7037,
        "sensor_34": 171.9375,
        "sensor_35": 341.9039,
        "sensor_36": 195.0655,
        "sensor_37": 90.32386,
        "sensor_38": 40.36458,
        "sensor_39": 31.51042,
        "sensor_40": 70.57291,
        "sensor_41": 30.98958,
        "sensor_42": 31.770832061767603,
        "sensor_43": 41.92708,
        "sensor_44": 39.6412,
        "sensor_45": 65.68287,
        "sensor_46": 50.92593,
        "sensor_47": 38.19444,
        "sensor_48": 157.9861,
        "sensor_49": 67.70834,
        "sensor_50": 243.0556,
        "sensor_51": 201.3889,
        "machine_status": "NORMAL"
    }
    
    # --- Parameters (should match the training configuration) ---
    window_size = 10      # The sliding window length used during training.
    hidden_size = 128
    num_layers = 3
    dropout = 0.3
    num_classes = 3       # Expected classes: "BROKEN", "NORMAL", "RECOVERING"
    
    # --- Preprocess the JSON input ---
    X_window, features = preprocess_json_data(json_input, window_size=window_size)
    # Convert the processed window into a PyTorch tensor.
    X_tensor = torch.tensor(X_window, dtype=torch.float32)
    
    # --- Load the Saved Model ---
    model_path = "/mnt/iusers01/fse-ugpgt01/compsci01/g49678gs/Hackafuture/final_model.pt"  # Ensure this file exists from training.
    input_size = X_tensor.shape[2]
    model = LSTMAttentionPredictor(input_size, hidden_size, num_layers, num_classes,
                                   dropout=dropout, use_mlp=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # --- Run Prediction ---
    with torch.no_grad():
        logits = model(X_tensor)
        pred = torch.argmax(logits, dim=1).item()
    
    # --- Map the Predicted Label to its Machine Status ---
    # (Assuming the label mapping from training is as follows:
    #   0 -> "BROKEN", 1 -> "NORMAL", 2 -> "RECOVERING")
    label_mapping = {0: "BROKEN", 1: "NORMAL", 2: "RECOVERING"}
    predicted_status = label_mapping.get(pred, "Unknown")
    
    # --- Create Output JSON ---
    output = {
        "device_id": json_input.get("device_id"),
        "timestamp": json_input.get("timestamp"),
        "predicted_machine_status": predicted_status,
        "predicted_label": pred
    }
    
    # Print the JSON output.
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
