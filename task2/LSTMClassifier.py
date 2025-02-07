from load_csv import *
from Imports import *

class EnhancedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super(EnhancedLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # LSTM layer with bidirectional option & dropout
        self.fc_input = nn.Linear(input_dim, hidden_dim)  # Convert input to LSTM size

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional)

        # Fully connected layers with BatchNorm & Dropout
        self.fc1 = nn.Linear(hidden_dim * num_directions, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)  # Final output layer

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_input(x).unsqueeze(1)  # Convert to LSTM format: (batch, seq_length=1, hidden_dim)
        _, (hidden, _) = self.lstm(x)  # Get hidden state

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate both directions
        else:
            hidden = hidden[-1]

        # Pass through FC layers with activation & dropout
        out = self.relu(self.bn1(self.fc1(hidden)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        return out

# Initialize model with improved parameters
INPUT_DIM = X_train.shape[1]  # Number of features
HIDDEN_DIM = 128  # Increased hidden units
NUM_LAYERS = 3  # More layers for deeper feature extraction
NUM_CLASSES = len(event_labels)  # Number of accident types
DROPOUT = 0.3  # Prevents overfitting
BIDIRECTIONAL = True  # Uses forward & backward information

model = EnhancedLSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT, BIDIRECTIONAL).to(device)

# Optimizer & Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # AdamW for better generalization
loss_function = nn.CrossEntropyLoss()  # Multi-class classification loss

# Mixed Precision Training Scaler
scaler = torch.GradScaler(device="cuda")
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")



















