import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Step 1: Transformer for Feature Extraction ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EEGTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.project = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x = self.project(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch_size, d_model)
        return x

# ========== Step 2: SVM Classifier ==========

def train_svm(features, labels):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ])
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01, 0.001]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(features, labels)
    return grid_search.best_estimator_

# ========== Step 3: Complete Training Pipeline ==========

def extract_transformer_features(model, eeg_data, device='cpu'):
    model.eval()
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        features = model(eeg_tensor).cpu().numpy()
    return features

# Example Usage
if __name__ == "__main__":
    # Simulated data (replace with real EEG data)
    N, T, C = 100, 200, 14  # N: trials, T: time, C: channels
    X = np.random.randn(N, T, C)  # EEG data
    y = np.random.randint(0, 2, N)  # Labels: 0 = non-P300, 1 = P300

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer = EEGTransformer(input_dim=C).to(device)

    # Convert to transformer input format
    features = extract_transformer_features(transformer, X, device=device)

    # Train SVM
    svm_model = train_svm(features, y)

    # Predict on the same features (for demonstration)
    predictions = svm_model.predict(features)
    print("Sample Predictions:", predictions[:10])
