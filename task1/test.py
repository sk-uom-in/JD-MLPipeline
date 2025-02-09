#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

This script loads the saved model (final_model.pt) and tests it on input data.
You can supply input in one of two modes:
  1. File mode: Use a CSV (e.g. pump_sensor.csv) that was used during training.
     This mode pre-processes the data into sliding windows, uses the test split,
     and prints for each test window:
       - Predicted machine status (decoded from labels)
       - Actual machine status (if available)
       - The time difference between the window’s end timestamp and the predicted event timestamp.
     It also reports overall accuracy.
  2. Manual mode: Supply a CSV file with sensor rows (a single sliding window)
     and the script prints the predicted machine status and (if timestamp info is available)
     the time until the predicted event.
     
No retraining occurs—the saved model is loaded and used directly for predictions.
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Data_prep import preprocess_data  # Uses the same preprocessing pipeline as during training

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


##########################################
# Model Definition (must match the one used during training)
##########################################
class LSTMAttentionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_mlp=True):
        """
        LSTM-Attention model for time-series classification.
        Args:
            input_size (int): Number of features per time step.
            hidden_size (int): LSTM hidden size.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            use_mlp (bool): Whether to use a deeper MLP block after attention.
        """
        super(LSTMAttentionPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_mlp = use_mlp

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism: a learnable attention vector.
        self.attention_vector = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))

        # MLP block (if enabled) for classification.
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
        # x: (batch, seq_length, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_length, hidden_size)

        # Compute attention scores
        energy = torch.tanh(lstm_out)            # (batch, seq_length, hidden_size)
        attn_scores = torch.matmul(energy, self.attention_vector)  # (batch, seq_length)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch, seq_length, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_size)
        context = self.dropout(context)

        # Classification block.
        if self.use_mlp:
            output = self.mlp(context)
        else:
            output = self.fc(context)
        return output


##########################################
# Helper Functions
##########################################
def load_model(input_size, num_classes, model_path="final_model.pt"):
    """
    Loads the saved model from disk.
    Adjust the hyperparameters if they differ from training.
    """
    hidden_size = 128
    num_layers = 3
    dropout = 0.3
    model = LSTMAttentionPredictor(input_size, hidden_size, num_layers, num_classes,
                                   dropout=dropout, use_mlp=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compute_window_timestamps(csv_path, window_size, lead_time):
    """
    Reads the original CSV (which must include a 'timestamp' column) and computes,
    for each sliding window, the window end timestamp and the predicted (future) timestamp.
    Returns a list of tuples: (window_end_timestamp, predicted_timestamp).
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    timestamps = []
    num_rows = len(df)
    max_start = num_rows - window_size - lead_time + 1
    for i in range(max_start):
        window_end_index = i + window_size - 1
        label_index = window_end_index + lead_time
        if label_index < num_rows:
            window_end_time = df.loc[window_end_index, 'timestamp']
            predicted_time = df.loc[label_index, 'timestamp']
            timestamps.append((window_end_time, predicted_time))
    return timestamps


def test_on_file(csv_path, window_size, lead_time, window_index):
    """
    Loads the full CSV, preprocesses it, loads the saved model, and
    makes a prediction on the specified test window.
    Also computes and prints timestamp information and prediction accuracy.
    """
    # Preprocess data (using same parameters as training)
    X_train, y_train, X_test, y_test, scaler, label_encoder, _ = preprocess_data(
        csv_path, window_size=window_size, lead_time=lead_time, test_split=0.2
    )
    total_test_windows = X_test.shape[0]
    if total_test_windows == 0:
        print("No test windows available. Check window_size, lead_time, or input CSV.")
        return

    if window_index < 0 or window_index >= total_test_windows:
        print(f"Invalid window index. Please choose an index between 0 and {total_test_windows - 1}.")
        return

    # Load the saved model.
    input_size = X_test.shape[2]
    num_classes = len(label_encoder.classes_)
    model = load_model(input_size, num_classes)

    # Get prediction for the specified test window.
    window_tensor = X_test[window_index].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(window_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    predicted_status = label_encoder.inverse_transform([pred_class])[0]
    actual_status = label_encoder.inverse_transform([y_test[window_index].item()])[0]
    print(f"\nTest Window Index: {window_index}")
    print(f"Predicted Machine Status: {predicted_status}")
    print(f"Actual Machine Status   : {actual_status}")

    # Compute timestamp info using the original CSV.
    try:
        timestamps = compute_window_timestamps(csv_path, window_size, lead_time)
        # Adjust window index for the full dataset (since train windows come first)
        original_index = X_train.shape[0] + window_index
        if original_index < len(timestamps):
            window_end_time, predicted_time = timestamps[original_index]
            time_delta = predicted_time - window_end_time
            print(f"Window End Time         : {window_end_time}")
            print(f"Predicted Event Time    : {predicted_time}")
            print(f"Time Until Event        : {time_delta}")
        else:
            print("Timestamp information not available for the selected window index.")
    except Exception as e:
        print("Error computing timestamp information:", e)

    # Also compute overall accuracy on the test set.
    all_preds = []
    with torch.no_grad():
        for i in range(total_test_windows):
            batch = X_test[i].unsqueeze(0).to(device)
            logits = model(batch)
            pred = torch.argmax(logits, dim=1).item()
            all_preds.append(pred)
    all_preds = np.array(all_preds)
    all_true = y_test.numpy()
    overall_accuracy = np.mean(all_preds == all_true)
    print(f"\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%")


def test_on_manual(manual_csv, window_size, lead_time):
    """
    Loads a manual input CSV file (with sensor rows) and makes a prediction.
    If the CSV contains a 'timestamp' column, timestamp info is also computed.
    """
    manual_df = pd.read_csv(manual_csv)
    # If manual CSV does not have a 'machine_status' column, add a dummy value.
    if 'machine_status' not in manual_df.columns:
        manual_df['machine_status'] = 'NORMAL'
    # Save to a temporary CSV so that we can use the same preprocessing pipeline.
    temp_csv = "temp_manual_input.csv"
    manual_df.to_csv(temp_csv, index=False)
    # Preprocess the manual input; use test_split=0.0 to get all windows.
    X_train_m, y_train_m, X_test_m, y_test_m, scaler_m, label_encoder_m, _ = preprocess_data(
        temp_csv, window_size=window_size, lead_time=lead_time, test_split=0.0
    )
    os.remove(temp_csv)

    if X_train_m.shape[0] > 0:
        window_tensor = X_train_m[0].unsqueeze(0).to(device)
    elif X_test_m.shape[0] > 0:
        window_tensor = X_test_m[0].unsqueeze(0).to(device)
    else:
        print("No valid sliding window could be created from the manual input.")
        return

    input_size = window_tensor.shape[2]
    num_classes = len(label_encoder_m.classes_)
    model = load_model(input_size, num_classes)
    with torch.no_grad():
        logits = model(window_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    predicted_status = label_encoder_m.inverse_transform([pred_class])[0]
    print(f"\nPredicted Machine Status (Manual Input): {predicted_status}")

    # If timestamp info is available, compute it.
    try:
        timestamps = compute_window_timestamps(manual_csv, window_size, lead_time)
        if len(timestamps) > 0:
            window_end_time, predicted_time = timestamps[0]
            time_delta = predicted_time - window_end_time
            print(f"Window End Time      : {window_end_time}")
            print(f"Predicted Event Time : {predicted_time}")
            print(f"Time Until Event     : {time_delta}")
        else:
            print("No timestamp information available in manual input.")
    except Exception as e:
        print("Error computing timestamp information for manual input:", e)
    print(f"This prediction is for {lead_time} minutes ahead of the provided data window.")


##########################################
# Main Routine
##########################################
def main():
    parser = argparse.ArgumentParser(
        description="Test the saved model (final_model.pt) on input data. "
                    "Choose 'file' mode to test on a CSV used during training or 'manual' mode to provide your own CSV."
    )
    parser.add_argument('--mode', type=str, choices=['file', 'manual'], default='file',
                        help="Choose 'file' to test on a CSV or 'manual' for manual input CSV.")
    parser.add_argument('--csv_path', type=str, default="pump_sensor.csv",
                        help="Path to the CSV file (for file mode).")
    parser.add_argument('--window_index', type=int, default=0,
                        help="Test window index to predict (for file mode).")
    parser.add_argument('--window_size', type=int, default=10,
                        help="Sliding window size (number of rows).")
    parser.add_argument('--lead_time', type=int, default=60,
                        help="Lead time (in minutes) for prediction.")
    parser.add_argument('--manual_csv', type=str,
                        help="Path to manual input CSV (for manual mode).")
    args = parser.parse_args()

    if args.mode == 'file':
        test_on_file(args.csv_path, args.window_size, args.lead_time, args.window_index)
    elif args.mode == 'manual':
        if not args.manual_csv:
            print("Please provide --manual_csv with the path to your manual input CSV.")
        else:
            test_on_manual(args.manual_csv, args.window_size, args.lead_time)
    else:
        print("Invalid mode selected. Choose 'file' or 'manual'.")


if __name__ == "__main__":
    main()
