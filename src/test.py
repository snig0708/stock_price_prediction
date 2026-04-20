import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================
# CONFIG
# ======================
DATA_PATH = "../data/stock_data.csv"   # change if needed
MODEL_PATH = "../saved_models/best_stock_model.pt"


# ======================
# DEVICE
# ======================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ======================
# LOAD CHECKPOINT
# ======================
checkpoint = torch.load(MODEL_PATH, map_location=device)

BEST_MODEL = checkpoint["model_name"]
TARGET_STOCK = checkpoint["target_stock"]
SEQ_LENGTH = checkpoint["seq_length"]
INPUT_SIZE = checkpoint["input_size"]
HIDDEN_SIZE = checkpoint["hidden_size"]
NUM_LAYERS = checkpoint["num_layers"]
feature_cols = checkpoint["feature_cols"]

print("Best model:", BEST_MODEL)
print("Target stock:", TARGET_STOCK)


# ======================
# LOAD DATA
# ======================
df_raw = pd.read_csv(DATA_PATH)

date_col = df_raw.columns[0]
df = df_raw.copy()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df = df.set_index(date_col)

series_df = df[[TARGET_STOCK]].rename(columns={TARGET_STOCK: "close"}).dropna().copy()


# ======================
# FEATURE ENGINEERING
# ======================
df_features = series_df.copy()

df_features["return_1d"] = df_features["close"].pct_change()
df_features["ma_5"] = df_features["close"].rolling(5).mean()
df_features["ma_20"] = df_features["close"].rolling(20).mean()
df_features["volatility_5"] = df_features["return_1d"].rolling(5).std()

df_features = df_features.dropna().copy()

target_idx = feature_cols.index("close")


# ======================
# SPLIT / SCALE
# ======================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

n = len(df_features)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df_features.iloc[:train_end].copy()
val_df = df_features.iloc[train_end:val_end].copy()
test_df = df_features.iloc[val_end:].copy()

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])


# ======================
# SEQUENCE CREATION
# ======================
def create_sequences(data, seq_length, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y).reshape(-1, 1)

X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH, target_idx=target_idx)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)


# ======================
# MODELS
# ======================
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


if BEST_MODEL == "RNN":
    model = RNNModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
elif BEST_MODEL == "LSTM":
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
else:
    model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()


# ======================
# HELPERS
# ======================
def inverse_transform_target(values, scaler, feature_count, target_idx):
    temp = np.zeros((len(values), feature_count))
    temp[:, target_idx] = values.reshape(-1)
    inv = scaler.inverse_transform(temp)
    return inv[:, target_idx].reshape(-1, 1)


def directional_accuracy_from_last_close(last_close, actual_next, pred_next):
    actual_move = actual_next.reshape(-1) - last_close.reshape(-1)
    pred_move = pred_next.reshape(-1) - last_close.reshape(-1)

    actual_dir = (actual_move > 0).astype(int)
    pred_dir = (pred_move > 0).astype(int)

    return (actual_dir == pred_dir).mean()


# ======================
# PREDICT / EVALUATE
# ======================
preds, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        preds.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

preds = np.array(preds)
actuals = np.array(actuals)

preds_inv = inverse_transform_target(preds, scaler, len(feature_cols), target_idx)
actuals_inv = inverse_transform_target(actuals, scaler, len(feature_cols), target_idx)

last_close_scaled = X_test[:, -1, target_idx].numpy().reshape(-1, 1)
last_close_inv = inverse_transform_target(last_close_scaled, scaler, len(feature_cols), target_idx)

mae = mean_absolute_error(actuals_inv, preds_inv)
rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))
dacc = directional_accuracy_from_last_close(last_close_inv, actuals_inv, preds_inv)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test Directional Accuracy: {dacc:.4f}")