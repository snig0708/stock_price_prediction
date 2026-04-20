import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================
# CONFIG
# ======================
DATA_PATH = "../data/stock_data.csv"   # change if needed
TARGET_STOCK = "Stock_1"                           # change if needed

SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SEED = 42


# ======================
# SETUP
# ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ======================
# LOAD DATA
# ======================
df_raw = pd.read_csv(DATA_PATH)
print("Raw shape:", df_raw.shape)

date_col = df_raw.columns[0]
df = df_raw.copy()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df = df.set_index(date_col)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_STOCK not in numeric_cols:
    raise ValueError(
        f"'{TARGET_STOCK}' not found. Available stock columns include: {numeric_cols[:10]}"
    )

series_df = df[[TARGET_STOCK]].rename(columns={TARGET_STOCK: "close"}).dropna().copy()

print("Using stock:", TARGET_STOCK)
print(series_df.head())


# ======================
# FEATURE ENGINEERING
# ======================
df_features = series_df.copy()

df_features["return_1d"] = df_features["close"].pct_change()
df_features["ma_5"] = df_features["close"].rolling(5).mean()
df_features["ma_20"] = df_features["close"].rolling(20).mean()
df_features["volatility_5"] = df_features["return_1d"].rolling(5).std()

df_features = df_features.dropna().copy()

feature_cols = ["close", "return_1d", "ma_5", "ma_20", "volatility_5"]
target_col = "close"
target_idx = feature_cols.index(target_col)

print("Feature columns:", feature_cols)
print("Prepared shape:", df_features.shape)


# ======================
# VISUALIZE
# ======================
plt.figure(figsize=(14, 5))
plt.plot(df_features.index, df_features["close"])
plt.title(f"{TARGET_STOCK} Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.show()


# ======================
# SPLIT
# ======================
n = len(df_features)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df_features.iloc[:train_end].copy()
val_df = df_features.iloc[train_end:val_end].copy()
test_df = df_features.iloc[val_end:].copy()

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)


# ======================
# SCALE
# ======================
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


X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH, target_idx=target_idx)
X_val, y_val = create_sequences(val_scaled, SEQ_LENGTH, target_idx=target_idx)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH, target_idx=target_idx)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)


# ======================
# TENSORS / LOADERS
# ======================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

INPUT_SIZE = X_train.shape[2]
HIDDEN_SIZE = 64
NUM_LAYERS = 2


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


# ======================
# HELPERS
# ======================
def inverse_transform_target(values, scaler, feature_count, target_idx):
    temp = np.zeros((len(values), feature_count))
    temp[:, target_idx] = values.reshape(-1)
    inv = scaler.inverse_transform(temp)
    return inv[:, target_idx].reshape(-1, 1)


def evaluate_regression(actuals, preds):
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return mae, rmse


def directional_accuracy_from_last_close(last_close, actual_next, pred_next):
    actual_move = actual_next.reshape(-1) - last_close.reshape(-1)
    pred_move = pred_next.reshape(-1) - last_close.reshape(-1)

    actual_dir = (actual_move > 0).astype(int)
    pred_dir = (pred_move > 0).astype(int)

    return (actual_dir == pred_dir).mean()


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict_model(model, data_loader):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            preds.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    return np.array(preds), np.array(actuals)


# ======================
# TRAIN ALL MODELS
# ======================
models = {
    "RNN": RNNModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS),
    "LSTM": LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS),
    "GRU": GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS),
}

trained_models = {}
history = {}
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LR
    )
    trained_models[model_name] = trained_model
    history[model_name] = {"train": train_losses, "val": val_losses}


# ======================
# EVALUATE MODELS
# ======================
last_close_scaled = X_test[:, -1, target_idx].numpy().reshape(-1, 1)
last_close = inverse_transform_target(last_close_scaled, scaler, len(feature_cols), target_idx)

for model_name, model in trained_models.items():
    preds_scaled, actuals_scaled = predict_model(model, test_loader)

    preds = inverse_transform_target(preds_scaled, scaler, len(feature_cols), target_idx)
    actuals = inverse_transform_target(actuals_scaled, scaler, len(feature_cols), target_idx)

    mae, rmse = evaluate_regression(actuals, preds)
    dacc = directional_accuracy_from_last_close(last_close, actuals, preds)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": dacc
    })

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\nModel comparison:")
print(results_df)


# ======================
# BASELINE
# ======================
actuals_baseline = inverse_transform_target(y_test.numpy(), scaler, len(feature_cols), target_idx)
preds_baseline = last_close.copy()

baseline_mae, baseline_rmse = evaluate_regression(actuals_baseline, preds_baseline)
baseline_dacc = directional_accuracy_from_last_close(last_close, actuals_baseline, preds_baseline)

baseline_df = pd.DataFrame([{
    "Model": "Naive Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "Directional_Accuracy": baseline_dacc
}])

print("\nBaseline:")
print(baseline_df)


# ======================
# PLOTS
# ======================
plt.figure(figsize=(14, 5))
for model_name, losses in history.items():
    plt.plot(losses["train"], label=f"{model_name} Train")
    plt.plot(losses["val"], label=f"{model_name} Val")
plt.title("Training / Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

preds_scaled, actuals_scaled = predict_model(best_model, test_loader)
preds = inverse_transform_target(preds_scaled, scaler, len(feature_cols), target_idx)
actuals = inverse_transform_target(actuals_scaled, scaler, len(feature_cols), target_idx)

print("\nBest model:", best_model_name)

plt.figure(figsize=(15, 5))
plt.plot(actuals[:200], label="Actual")
plt.plot(preds[:200], label=f"{best_model_name} Predicted")
plt.title(f"{TARGET_STOCK}: Actual vs Predicted ({best_model_name})")
plt.xlabel("Time Step")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()


# ======================
# SAVE BEST MODEL
# ======================
os.makedirs("../saved_models", exist_ok=True)

torch.save(
    {
        "model_name": best_model_name,
        "state_dict": best_model.state_dict(),
        "feature_cols": feature_cols,
        "target_stock": TARGET_STOCK,
        "seq_length": SEQ_LENGTH,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    },
    "../saved_models/best_stock_model.pt"
)

print("\nSaved best model to ../saved_models/best_stock_model.pt")