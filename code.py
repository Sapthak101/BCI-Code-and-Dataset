import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# EEG Transformer Model
class EEGTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, max_len=100):
        super(EEGTransformer, self).__init__()
        self.linear_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.linear_proj(x)
        x = x + self.pos_embedding[:, :x.size(1)]
        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        return x

# Preprocessing EEG
def preprocess_eeg(eeg_data, fs=256.0):
    def bandpass_filter(data, lowcut=0.1, highcut=15.0, fs=256.0, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    filtered = np.array([[bandpass_filter(ch, fs=fs) for ch in trial.T] for trial in eeg_data])
    normalized = (filtered - filtered.mean(axis=-1, keepdims=True)) / filtered.std(axis=-1, keepdims=True)
    return normalized.transpose(0, 2, 1)

# Train SVM
def train_svm(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    parameters = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    clf = GridSearchCV(SVC(kernel='rbf'), parameters)
    clf.fit(features_scaled, labels)
    return clf, scaler

# Predict SVM
def predict_svm(clf, scaler, test_features):
    test_features_scaled = scaler.transform(test_features)
    return clf.predict(test_features_scaled)
