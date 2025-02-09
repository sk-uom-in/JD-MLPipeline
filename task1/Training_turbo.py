import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ================================
# 1. Load Training Data and Compute Scaling Factor
# ================================
train_csv = "/mnt/iusers01/fse-ugpgt01/compsci01/g49678gs/Hackafuture/Turbo_prediction/Data/FD_001_train_processed.csv"  # Preprocessed CSV with RUL column
df_train = pd.read_csv(train_csv)
print("Loaded training data shape:", df_train.shape)

# Compute maximum RUL from training data (for normalization)
max_rul = df_train['RUL'].max()
print("Maximum RUL in training data:", max_rul)

# Define sensor and target columns (adjust if needed)
sensor_cols = [
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',
    'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
]
target_col = 'RUL'

# ================================
# 2. Create a Dataset with Normalized Targets
# ================================
class FDTimeSeriesDataset(Dataset):
    """
    Dataset for FD_001 that groups data by engine (unit_ID) and creates overlapping sliding windows.
    Each sample is a window of sensor data and its target is the normalized RUL (i.e. original RUL divided by target_norm)
    at the window's last time step.
    """
    def __init__(self, dataframe, sensor_cols, target_col, window_length=30, shift=1, target_norm=1.0):
        self.window_length = window_length
        self.shift = shift
        self.sensor_cols = sensor_cols
        self.target_col = target_col
        self.target_norm = target_norm  # e.g., max_rul
        
        # Sort data by engine (unit_ID) and cycle
        df_sorted = dataframe.sort_values(by=['unit_ID', 'cycles']).copy()
        
        self.windows = []
        self.targets = []
        for unit, group in df_sorted.groupby('unit_ID'):
            group = group.reset_index(drop=True)
            sensor_data = group[sensor_cols].values.astype(np.float32)
            rul_data = group[target_col].values.astype(np.float32)
            n_steps = sensor_data.shape[0]
            if n_steps >= window_length:
                num_windows = (n_steps - window_length) // shift + 1
                for i in range(num_windows):
                    start = i * shift
                    end = start + window_length
                    self.windows.append(sensor_data[start:end])
                    # Normalize target: divide by target_norm
                    self.targets.append(rul_data[end - 1] / self.target_norm)
                    
        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        print(f"Dataset created with {len(self.windows)} samples. Each sample shape: {self.windows.shape[1:]}")
    
    def __len__(self):
        return self.windows.shape[0]
    
    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]

# Create the dataset (using target_norm = max_rul)
WINDOW_LENGTH = 30
dataset = FDTimeSeriesDataset(df_train, sensor_cols, target_col, window_length=WINDOW_LENGTH, shift=1, target_norm=max_rul)

# Split dataset into training and validation sets (80/20 split)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 128
num_workers = 8  # Adjust as appropriate for your CPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ================================
# 3. Define the LSTM-Based RUL Prediction Model
# ================================
class LSTMRULModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, mlp_dim, dropout):
        """
        Args:
            input_dim: Number of sensor features per time step.
            hidden_dim: Hidden dimension for the LSTM.
            num_layers: Number of LSTM layers.
            mlp_dim: Hidden dimension for the MLP head.
            dropout: Dropout probability (applied in LSTM and MLP for regularization/Bayesian uncertainty).
        """
        super(LSTMRULModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Use the output from the last time step
        last_out = self.dropout(last_out)
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # (batch,)

input_dim = len(sensor_cols)
hidden_dim = 128
num_layers = 3
mlp_dim = 128
dropout = 0.3
model = LSTMRULModel(input_dim, hidden_dim, num_layers, mlp_dim, dropout)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0031278668112885467, weight_decay=1.0269626404425149e-05)

# ================================
# 4. Train the Model
# ================================
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    val_rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val RMSE (norm) = {val_rmse:.4f}")
    
# Save the final model
torch.save(model.state_dict(), "best_lstm_rul_model.pt")
print("Final model training complete.")

# ================================
# 5. Evaluate on Test Data Using the Separate RUL File
# ================================
# Load test CSV (which has the same columns as training except RUL)
test_csv = "/mnt/iusers01/fse-ugpgt01/compsci01/g49678gs/Hackafuture/Turbo_prediction/Data/test_001_train_processed.csv"
df_test = pd.read_csv(test_csv)
print("Loaded test data shape:", df_test.shape)

# Load the separate RUL file; if it contains a header like "RUL", set header=0.
df_test_RUL = pd.read_csv("/mnt/iusers01/fse-ugpgt01/compsci01/g49678gs/Hackafuture/Turbo_prediction/Data/rul_001_train_processed.csv", header=0)
true_rul_array = df_test_RUL['RUL'].astype(float).values
print("Loaded ground truth RUL entries:", len(true_rul_array))

# Ensure that the number of unique test engines matches the number of RUL entries:
unique_units = sorted(df_test['unit_ID'].unique())
num_units = len(unique_units)
min_units = min(num_units, len(true_rul_array))
if num_units != len(true_rul_array):
    print(f"Mismatch detected: {num_units} engines in test CSV vs. {len(true_rul_array)} RUL entries.")
    print(f"Using the first {min_units} engines for evaluation.")
    unique_units = unique_units[:min_units]
true_rul = true_rul_array[:min_units]

# (Optional) Inspect the final window statistics for one engine
example_unit = unique_units[0]
example_group = df_test[df_test['unit_ID'] == example_unit].sort_values(by='cycles')
print(f"\nEngine {example_unit} final window (last {WINDOW_LENGTH} cycles):")
print(example_group[sensor_cols].tail(WINDOW_LENGTH).describe())

# Predict for each engine using the final available window, then re-scale predictions
model.eval()
engine_predictions = {}
for unit in unique_units:
    group = df_test[df_test['unit_ID'] == unit].sort_values(by='cycles')
    if len(group) >= WINDOW_LENGTH:
        window = group[sensor_cols].values[-WINDOW_LENGTH:]
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(window_tensor)  # Prediction in normalized scale
        # Multiply by max_rul to recover original scale:
        engine_predictions[unit] = pred_norm.item() * max_rul
    else:
        print(f"Engine {unit} has less than {WINDOW_LENGTH} cycles; skipping.")

predicted_rul = np.array([engine_predictions[unit] for unit in unique_units])
print("\nPredicted RUL for test engines:")
for unit, pred in zip(unique_units, predicted_rul):
    print(f"Engine {unit}: Predicted RUL = {pred:.2f}")

# Compute RMSE using the original scale
test_rmse = np.sqrt(np.mean((predicted_rul - true_rul)**2))
print(f"\nTest RMSE: {test_rmse:.4f}")
