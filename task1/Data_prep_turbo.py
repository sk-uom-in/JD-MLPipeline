import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. Load the Preprocessed CSV Data
# ============================================================
csv_file = "/mnt/iusers01/fse-ugpgt01/compsci01/g49678gs/Hackafuture/Turbo_prediction/Data/FD_001_train_processed.csv"  # This is the CSV you previously saved.
df = pd.read_csv(csv_file)
print("Loaded data shape:", df.shape)
print(df.head())

# ============================================================
# 2. Define Columns to Use
# ============================================================
# In this example, we use sensor readings as features.
# You can adjust the list below if you want to include additional columns.
meta_cols = ['unit_ID', 'cycles', 'setting_1', 'setting_2', 'setting_3']
sensor_cols = [
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',
    'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
]
target_col = 'RUL'

# ============================================================
# 3. Scale Sensor Features
# ============================================================
# We scale the sensor data to improve training stability.
scaler = StandardScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols].values)
print("Sensor features scaled.")

# ============================================================
# 4. Create a PyTorch Dataset with Sliding Windows
# ============================================================
class FDTimeSeriesDataset(Dataset):
    """
    Converts the preprocessed FD_001 CSV (which contains time-series data for each engine)
    into a PyTorch dataset using sliding windows. For each engine (grouped by unit_ID),
    the data are segmented into overlapping windows of fixed length. The target for each window
    is the RUL at the last time step of that window.
    """
    def __init__(self, dataframe, sensor_cols, target_col, window_length=30, shift=1):
        self.window_length = window_length
        self.shift = shift
        self.sensor_cols = sensor_cols
        self.target_col = target_col
        
        # Ensure data is sorted by engine and cycle
        df_sorted = dataframe.sort_values(by=['unit_ID', 'cycles']).copy()
        
        self.windows = []
        self.targets = []
        
        # Process each engine unit separately
        for unit, group in df_sorted.groupby('unit_ID'):
            group = group.reset_index(drop=True)
            sensor_data = group[sensor_cols].values.astype(np.float32)  # shape: (n_steps, num_sensors)
            rul_data = group[target_col].values.astype(np.float32)         # shape: (n_steps,)
            n_steps = sensor_data.shape[0]
            # Only use engines with enough data for one window
            if n_steps >= window_length:
                num_windows = (n_steps - window_length) // shift + 1
                for i in range(num_windows):
                    start = i * shift
                    end = start + window_length
                    self.windows.append(sensor_data[start:end])
                    # The target is the RUL at the last time step of the window
                    self.targets.append(rul_data[end - 1])
                    
        # Convert the lists to torch tensors
        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        print(f"PyTorch Dataset created with {len(self.windows)} samples. Each sample shape: {self.windows.shape[1:]}")
    
    def __len__(self):
        return self.windows.shape[0]
    
    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]

# ============================================================
# The preprocessed dataset is now ready to be fed into your model.
# You can use torch.nn.DataParallel or DistributedDataParallel
# to leverage your two GPUs during training.
# ============================================================
