from Imports import *

# Path to operation data
data_path = "NuclearPowerPlantAccidentData/Operation_csv_data/"

# List all events (scenarios)
events = os.listdir(data_path)
# print(f"events: {events}")

# Assign numerical labels to each event type
event_labels = {event: idx for idx, event in enumerate(events)}
print(f"event_labels: {event_labels}")


# Load all event data with labels
def load_data_with_labels(data_path, event_labels):
    sequences = []
    labels = []

    for event, label in event_labels.items():
        event_path = os.path.join(data_path, event)
        a = 0
        # Get list of CSV files in the folder
        csv_files = [f for f in os.listdir(event_path) if f.endswith(".csv")]
        # print("label: ", label)
        for j in csv_files:
            
            a+=1
            # Pick only the first CSV file from the folder
            selected_file = j  # Change this to `random.choice(csv_files)` if you want random selection
            # print(event_path)
            # print("file: ", selected_file)
            
            file_path = os.path.join(event_path, selected_file)
            # print(file_path)
            df = pd.read_csv(file_path)

            # Remove "TIME" column
            if "TIME" in df.columns:
                df = df.drop(columns=["TIME"])

            # print(df.shape)
            if df.shape[0] < 300:
                continue
            # **Ensure only 96 columns are taken**
            df = df.iloc[:300, :96] 

            # Remove columns with NaNs
            df = df.dropna(axis=1)

            # Normalize features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(df)

            # print("[DEBUG] labels: ",label)


            # Create sequences for this file
            # for i in range(len(features_scaled) - seq_length):
            #     sequences.append(features_scaled[i:i + seq_length])
            #     labels.append(label)  # Assign the incident type label

            # Create sequences for this file
            for i in range(len(features_scaled)):
                sequences.append(features_scaled[i])
                labels.append(label)  # Assign the incident type label


            # print("[DEBUG] sequences shape: ",np.array(sequences).shape)
            # print("[DEBUG] sequences: ",np.array(sequences))
            # print("[DEBUG] labels shape: ",np.array(labels).shape)
            # print("[DEBUG] labels: ",label)
            # time.wait(10)
            # print("a: ",a)
            if a>0:
                break

    return np.array(sequences), np.array(labels)


# SEQ_LENGTH = 30

# Create sequences
X, y = load_data_with_labels(data_path, event_labels)

# print(X.shape)
# print(y.shape)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Classification labels need long type

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Efficient DataLoader with GPU optimization
BATCH_SIZE = 1024 # Increase batch size for faster training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
