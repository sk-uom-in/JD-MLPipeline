#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined training and testing script for the LSTM-Attention model.
This script:
1. Preprocesses the data from a CSV file (dropping columns with >70% missing values,
    filling missing numeric sensor readings, filtering to valid machine statuses, scaling, and creating sliding windows).
2. Trains an LSTM-attention model (with an optional MLP block) on the training set.
3. Saves the trained model.
4. Loads the saved model.
5. Evaluates the model on test data and calculates test accuracy.
6. Plots:
    a. Frequency distribution (bar plot) of predicted vs. actual classes.
    b. A time-series line plot of predicted and actual values (for a subset of test samples).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from Data_prep import *

# Set device (GPU if available)
device = 0
device1=1
print("Using device:", device, " ", device1)


class LSTMAttentionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, use_mlp=True):
        """
        Args:
            input_size (int): Number of features per time step.
            hidden_size (int): Number of neurons in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            use_mlp (bool): Whether to use a deeper MLP block after attention.
        """
        super(LSTMAttentionPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_mlp = use_mlp
        
        # Stacked LSTM: using num_layers parameter.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism: learnable attention vector.
        self.attention_vector = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        
        # Deeper MLP block: two hidden layers before final classification.
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
        # x: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        # Initialize hidden and cell states.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, hidden_size)
        # Attention: compute scores and weighted context.
        energy = torch.tanh(lstm_out)  # (batch_size, seq_length, hidden_size)
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
    # 3. Training Function
    ############################################

def train_model(model, train_loader, num_epochs, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for the specified number of epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device1)
            y_batch = y_batch.to(device1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        if scheduler:
            scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model


############################################
# 4. Main Training & Testing Block
############################################
# Parameters (adjust as needed)
filepath = "pump_sensor.csv"  # CSV file with sensor data.
window_size = 10              # Input sequence length (in minutes/rows).
lead_time = 60                # Predict status 60 minutes ahead.
num_epochs = 50               # Number of training epochs.
batch_size = 256               # Batch size.
hidden_size = 128             # LSTM hidden size.
num_layers = 3                # Number of stacked LSTM layers (increased from 2).
dropout = 0.3                 # Dropout probability.

############################################
# Preprocess the Data
############################################
X_train, y_train, X_test, y_test, scaler, label_encoder, df_processed = preprocess_data(
    filepath, window_size=window_size, lead_time=lead_time, test_split=0.2
)

# Create DataLoader for training set.
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Determine input feature dimension and number of classes.
num_features = X_train.shape[2]
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}, Input feature dimension: {num_features}")

############################################
# Initialize and Train the Model
############################################
model = LSTMAttentionPredictor(
    input_size=num_features,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout,
    use_mlp=True  # Using deeper MLP block.
).to(device1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

print("Starting training...")
trained_model = train_model(model, train_loader, num_epochs, optimizer, criterion, device1, scheduler)

# Save the trained model.
torch.save(trained_model.state_dict(), "final_model2.pt")
print("Model training complete. Model saved as 'final_model.pt'.")

############################################
# Evaluate the Model on Test Data
############################################
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# (Optional) Load the saved model state.
trained_model.load_state_dict(torch.load("final_model2.pt", map_location=device))
trained_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = trained_model(X_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate test accuracy.
accuracy = np.mean(all_preds == all_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

############################################
# Plot Frequency Distribution: Actual vs. Predicted
############################################
unique_classes = np.unique(all_labels)
counts_actual = [np.sum(all_labels == cls) for cls in unique_classes]
counts_predicted = [np.sum(all_preds == cls) for cls in unique_classes]

x = np.arange(len(unique_classes))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, counts_actual, width, label='Actual')
rects2 = ax.bar(x + width/2, counts_predicted, width, label='Predicted')

ax.set_ylabel('Frequency')
ax.set_title('Frequency Distribution: Actual vs. Predicted Classes')
ax.set_xticks(x)
ax.set_xticklabels([f'Class {int(cls)}' for cls in unique_classes])
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()

# ############################################
# # Plot Time-Series: Predicted vs. Actual (subset)
# ############################################
# num_points = min(100, len(all_labels))
# plt.figure(figsize=(12, 6))
# plt.plot(range(num_points), all_labels[:num_points], marker='o', label='Actual')
# plt.plot(range(num_points), all_preds[:num_points], marker='x', label='Predicted')
# plt.xlabel('Test Sample Index')
# plt.ylabel('Class Label')
# plt.title('Predicted vs. Actual Machine Status Over Test Samples')
# plt.legend()
# plt.grid(True)
# plt.show(block=True)
# plt.savefig("plot_output.png")
# print("Plot saved as 'plot_output.png'")
