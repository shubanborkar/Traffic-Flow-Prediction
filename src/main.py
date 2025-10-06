import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
import pickle
import time
import psutil

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Memory check function (optional, for debugging)
def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}% used, {memory.available / 1024**2:.2f} MB available")

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# ============= DATA LOADING AND PREPROCESSING =============

def load_metr_la_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    print("Available keys in .npz file:", data.files)
    
    if 'data' in data:
        raw_data = data['data']
        timestamps = raw_data[:, 0]
        traffic_data = raw_data[:, 1:]
        
        try:
            traffic_data = traffic_data.astype(float)
        except ValueError as e:
            print("Error converting traffic data to float:", e)
            print("Sample of traffic_data:", traffic_data[:5])
            raise
        
        max_samples = min(10000, traffic_data.shape[0])
        if traffic_data.shape[0] > max_samples:
            print(f"Using {max_samples} samples to reduce computation time")
            traffic_data = traffic_data[:max_samples]
            timestamps = timestamps[:max_samples]
        
        print("Shape of traffic_data:", traffic_data.shape)
        print("Sample timestamps:", timestamps[:5])
        print("Sample traffic_data:", traffic_data[:5])
        return traffic_data
    else:
        print("Warning: Expected 'data' field not found. Available fields:", data.files)
        return data[data.files[0]]

def load_pems_bay_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    print("Available keys in .npz file:", data.files)
    
    if 'data' in data:
        raw_data = data['data']
        timestamps = raw_data[:, 0]
        traffic_data = raw_data[:, 1:]
        
        try:
            traffic_data = traffic_data.astype(float)
        except ValueError as e:
            print("Error converting traffic data to float:", e)
            print("Sample of traffic_data:", traffic_data[:5])
            raise
        
        max_samples = min(10000, traffic_data.shape[0])
        if traffic_data.shape[0] > max_samples:
            print(f"Using {max_samples} samples to reduce computation time")
            traffic_data = traffic_data[:max_samples]
            timestamps = timestamps[:max_samples]
        
        print("Shape of traffic_data:", traffic_data.shape)
        return traffic_data
    else:
        print("Warning: Expected 'data' field not found. Available fields:", data.files)
        return data[data.files[0]]

def preprocess_data(data, seq_length=12, pred_length=3):
    if len(data.shape) != 2:
        raise ValueError(f"Expected data shape [num_timesteps, num_sensors], got {data.shape}")
    
    num_timesteps, num_sensors = data.shape
    X, y = [], []
    for i in range(num_timesteps - seq_length - pred_length + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length].flatten())
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def normalize_data(train_data, val_data, test_data):
    scaler = StandardScaler()
    
    train_shape = train_data.shape
    val_shape = val_data.shape
    test_shape = test_data.shape
    
    train_data_flat = train_data.reshape(-1, train_data.shape[-1])
    scaler.fit(train_data_flat)
    
    train_data_scaled = scaler.transform(train_data_flat).reshape(train_shape)
    val_data_flat = val_data.reshape(-1, val_data.shape[-1])
    val_data_scaled = scaler.transform(val_data_flat).reshape(val_shape)
    test_data_flat = test_data.reshape(-1, test_data.shape[-1])
    test_data_scaled = scaler.transform(test_data_flat).reshape(test_shape)
    
    return train_data_scaled, val_data_scaled, test_data_scaled, scaler

# ============= MODEL DEFINITIONS =============

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = 32
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = 32
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = 32
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_length):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.nhead = nhead
        self.d_model = 32
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.output_layer(x)

# ============= TRAINING AND EVALUATION FUNCTIONS =============

def save_model(model, model_name, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if isinstance(model, nn.Module):
        save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_name': model.__class__.__name__,
            'model_config': {
                'input_dim': model.rnn.input_size if hasattr(model, 'rnn') else
                             model.lstm.input_size if hasattr(model, 'lstm') else
                             model.gru.input_size if hasattr(model, 'gru') else
                             model.input_proj.in_features,
                'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
                'output_dim': model.fc.out_features if hasattr(model, 'fc') else
                              model.output_layer.out_features,
                'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
                'd_model': model.d_model if hasattr(model, 'd_model') else None,
                'nhead': model.nhead if hasattr(model, 'nhead') else None,
                'seq_length': 12
            },
            'timestamp': timestamp
        }, save_path)
    else:
        save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to {save_path}")
    return save_path

def load_model(model_path, device=device):
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint['class_name'] == 'RNNModel':
            model = RNNModel(
                checkpoint['model_config']['input_dim'],
                checkpoint['model_config']['hidden_dim'],
                checkpoint['model_config']['output_dim'],
                checkpoint['model_config']['num_layers']
            )
        elif checkpoint['class_name'] == 'LSTMModel':
            model = LSTMModel(
                checkpoint['model_config']['input_dim'],
                checkpoint['model_config']['hidden_dim'],
                checkpoint['model_config']['output_dim'],
                checkpoint['model_config']['num_layers']
            )
        elif checkpoint['class_name'] == 'GRUModel':
            model = GRUModel(
                checkpoint['model_config']['input_dim'],
                checkpoint['model_config']['hidden_dim'],
                checkpoint['model_config']['output_dim'],
                checkpoint['model_config']['num_layers']
            )
        elif checkpoint['class_name'] == 'TransformerModel':
            model = TransformerModel(
                checkpoint['model_config']['input_dim'],
                checkpoint['model_config']['d_model'],
                checkpoint['model_config']['nhead'],
                checkpoint['model_config']['num_layers'],
                checkpoint['model_config']['output_dim'],
                checkpoint['model_config']['seq_length']
            )
        else:
            raise ValueError(f"Unknown model class: {checkpoint['class_name']}")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
    
    elif model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    else:
        raise ValueError(f"Unknown model format: {model_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=2, device=device):
    model.to(device)
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, model

def evaluate_model(model, test_loader, criterion=None, device=device, scaler=None, pred_length=None, num_sensors=None):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    if scaler is not None:
        predictions_inv = scaler.inverse_transform(predictions)
        actuals_inv = scaler.inverse_transform(actuals)
    else:
        predictions_inv = predictions
        actuals_inv = actuals
    
    mae = mean_absolute_error(actuals_inv, predictions_inv)
    rmse = np.sqrt(mean_squared_error(actuals_inv, predictions_inv))
    r2 = r2_score(actuals_inv, predictions_inv)
    
    if pred_length is not None and num_sensors is not None:
        predictions_reshaped = predictions_inv.reshape(-1, pred_length, num_sensors)
        actuals_reshaped = actuals_inv.reshape(-1, pred_length, num_sensors)
    else:
        predictions_reshaped = predictions_inv
        actuals_reshaped = actuals_inv
    
    if criterion is not None:
        return test_loss/len(test_loader), mae, rmse, r2, predictions_reshaped, actuals_reshaped
    else:
        return mae, rmse, r2, predictions_reshaped, actuals_reshaped

def train_svm_model(X_train, y_train):
    print("Starting SVM training...")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    max_samples = min(500, X_train_reshaped.shape[0])
    if X_train_reshaped.shape[0] > max_samples:
        print(f"Using {max_samples} samples for SVM training due to computational constraints")
        indices = np.random.choice(X_train_reshaped.shape[0], max_samples, replace=False)
        X_train_subset = X_train_reshaped[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_reshaped
        y_train_subset = y_train
    
    svm_model = MultiOutputRegressor(SVR(kernel='linear', C=1.0, gamma='scale'))
    svm_model.fit(X_train_subset, y_train_subset)
    print("SVM training completed")
    return svm_model

def evaluate_svm_model(svm_model, X_test, y_test, scaler=None):
    print("Starting SVM evaluation...")
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    max_samples = min(500, X_test_reshaped.shape[0])
    if X_test_reshaped.shape[0] > max_samples:
        print(f"Using {max_samples} samples for SVM evaluation due to computational constraints")
        indices = np.random.choice(X_test_reshaped.shape[0], max_samples, replace=False)
        X_test_subset = X_test_reshaped[indices]
        y_test_subset = y_test[indices]
    else:
        X_test_subset = X_test_reshaped
        y_test_subset = y_test
    
    y_pred = svm_model.predict(X_test_subset)
    
    if scaler is not None:
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test_subset)
    else:
        y_pred_inv = y_pred
        y_test_inv = y_test_subset
    
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    print("SVM evaluation completed")
    return mae, rmse, r2, y_pred_inv, y_test_inv

def train_logistic_model(X_train, y_train, threshold=None):
    print("Starting Logistic Regression training...")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    max_samples = min(1000, X_train_reshaped.shape[0])
    if X_train_reshaped.shape[0] > max_samples:
        print(f"Using {max_samples} samples for Logistic Regression training due to computational constraints")
        indices = np.random.choice(X_train_reshaped.shape[0], max_samples, replace=False)
        X_train_subset = X_train_reshaped[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_reshaped
        y_train_subset = y_train
    
    if threshold is None:
        threshold = np.median(y_train_subset)
    
    y_train_binary = (y_train_subset > threshold).astype(int)
    y_train_binary_first = y_train_binary[:, 0]
    
    log_model = LogisticRegression(max_iter=500, random_state=42, solver='liblinear')
    log_model.fit(X_train_subset, y_train_binary_first)
    print("Logistic Regression training completed")
    return log_model, threshold

def evaluate_logistic_model(log_model, X_test, y_test, threshold):
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    y_test_binary = (y_test > threshold).astype(int)
    y_test_binary_first = y_test_binary[:, 0]
    
    y_pred_prob = log_model.predict_proba(X_test_reshaped)[:, 1]
    y_pred = log_model.predict(X_test_reshaped)
    
    accuracy = accuracy_score(y_test_binary_first, y_pred)
    auc_score = roc_auc_score(y_test_binary_first, y_pred_prob)
    return accuracy, auc_score, y_pred, y_test_binary_first

# ============= VISUALIZATION FUNCTIONS =============

def plot_training_history(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pd.DataFrame({'Train Loss': train_losses, 'Validation Loss': val_losses}), palette='deep')
    plt.title(f'{model_name} - Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss_history.png'))
    plt.show()

def plot_predictions(predictions, actuals, model_name, n_samples=5, pred_length=3, num_sensors=207):
    """
    Plot predicted vs. actual traffic flow for multiple samples, sensors, and time steps.
    Shows all 3 prediction steps for up to 5 sensors per sample.
    """
    plt.figure(figsize=(15, 10))  # Larger figure for multiple subplots
    for sample_idx in range(min(n_samples, predictions.shape[0])):
        for sensor_idx in range(min(5, num_sensors)):  # Plot up to 5 sensors per sample
            plt.subplot(n_samples, 5, sample_idx * 5 + sensor_idx + 1)
            # Plot all 3 prediction steps for this sensor
            time_steps = np.arange(pred_length)
            sns.lineplot(x=time_steps, y=actuals[sample_idx, :, sensor_idx], label='Actual', color='blue', marker='o')
            sns.lineplot(x=time_steps, y=predictions[sample_idx, :, sensor_idx], label='Predicted', color='red', marker='o')
            plt.title(f'Sample {sample_idx + 1}, Sensor {sensor_idx + 1}')
            plt.xlabel('Time Step (5-min intervals)')
            plt.ylabel('Traffic Flow')
            if sample_idx == 0 and sensor_idx == 0:
                plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_traffic_flow_predictions.png'))
    plt.show()

def plot_comparison(results):
    models_with_rmse = [model for model in results.keys() if 'rmse' in results.get(model, {})]
    rmse_values = [results[model]['rmse'] for model in models_with_rmse]
    # Only include models with 'r2' in results
    models_with_r2 = [model for model in results.keys() if 'r2' in results.get(model, {})]
    r2_values = [results[model]['r2'] for model in models_with_r2 if 'r2' in results.get(model, {})]
    models_with_mae = [model for model in results.keys() if 'mae' in results.get(model, {})]
    mae_values = [results[model]['mae'] for model in models_with_mae]
    
    plt.figure(figsize=(12, 4))
    if models_with_rmse:
        plt.subplot(1, 3, 1)
        sns.barplot(x=models_with_rmse, y=rmse_values, palette='Blues_d')
        plt.title('RMSE Comparison (lower is better)')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45, fontsize=8)
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=6)
    
    if models_with_r2:
        plt.subplot(1, 3, 2)
        sns.barplot(x=models_with_r2, y=r2_values, palette='Greens_d')
        plt.title('R² Comparison (higher is better)')
        plt.ylabel('R²')
        plt.xticks(rotation=45, fontsize=8)
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=6)
    
    if models_with_mae:
        plt.subplot(1, 3, 3)
        sns.barplot(x=models_with_mae, y=mae_values, palette='Reds_d')
        plt.title('MAE Comparison (lower is better)')
        plt.ylabel('MAE')
        plt.xticks(rotation=45, fontsize=8)
        for i, v in enumerate(mae_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=6)
    
    plt.tight_layout()
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.show()

def plot_logistic_regression(logistic_preds, logistic_actuals, threshold, n_samples=5, model_name='Logistic Regression'):
    """
    Visualize Logistic Regression binary predictions and probabilities for traffic flow classification.
    Plots binary predictions, actual labels, and probabilities for a subset of samples.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot binary predictions vs. actual labels for up to n_samples
    for i in range(min(n_samples, len(logistic_preds))):
        plt.subplot(n_samples, 2, 2*i + 1)
        # Create a simple x-axis (sample index or time step)
        x = np.arange(len(logistic_preds[i:i+1]))  # Since logistic_preds is 1D, use a single value per sample
        sns.scatterplot(x=x, y=logistic_actuals[i], color='blue', label='Actual (0/1)', s=100, marker='o')
        sns.scatterplot(x=x, y=logistic_preds[i], color='red', label='Predicted (0/1)', s=100, marker='x')
        plt.title(f'Sample {i + 1} - Binary Predictions')
        plt.xlabel('Sample Index')
        plt.ylabel('Binary Value (0/1)')
        plt.legend()
        plt.grid(True)
        
        # Plot probabilities for the same sample
        plt.subplot(n_samples, 2, 2*i + 2)
        sns.barplot(x=x, y=logistic_preds[i:i+1], color='green', label='Probability (Class 1)')
        plt.title(f'Sample {i + 1} - Predicted Probability')
        plt.xlabel('Sample Index')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_logistic_regression_visualization.png'))
    plt.show()

# ============= MAIN FUNCTION =============

def main(dataset_path, dataset_type='metr_la', load_saved_models=False):
    print(f"Starting traffic flow prediction pipeline using {dataset_type} dataset")
    
    if dataset_type == 'metr_la':
        data = load_metr_la_data(dataset_path)
    elif dataset_type == 'pems_bay':
        data = load_pems_bay_data(dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Data loaded with shape: {data.shape}")
    
    seq_length = 12
    pred_length = 3
    num_sensors = data.shape[1]
    
    X, y = preprocess_data(data, seq_length, pred_length)
    print(f"Prepared sequences X: {X.shape}, y: {y.shape}")
    
    max_samples = min(5000, X.shape[0])
    if X.shape[0] > max_samples:
        print(f"Using {max_samples} samples for training to reduce computation time")
        X = X[:max_samples]
        y = y[:max_samples]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train, X_val, X_test, scaler_X = normalize_data(X_train, X_val, X_test)
    y_train, y_val, y_test, scaler_y = normalize_data(y_train, y_val, y_test)
    
    batch_size = 128
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), 
                           batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), 
                            batch_size=batch_size)
    
    results = {}
    
    # 3. Train and evaluate RNN model
    if not load_saved_models or not os.path.exists('models/rnn_model.pth'):
        print("\n=== Training RNN Model ===")
        input_dim = X_train.shape[2]
        hidden_dim = 32
        output_dim = y_train.shape[1]
        num_layers = 1
        
        rnn_model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
        
        train_losses, val_losses, rnn_model = train_model(
            rnn_model, train_loader, val_loader, criterion, optimizer
        )
        rnn_model_path = save_model(rnn_model, 'rnn_model')
        plot_training_history(train_losses, val_losses, 'RNN')
    else:
        print("\n=== Loading saved RNN Model ===")
        rnn_model_path = 'models/rnn_model.pth'
        rnn_model = load_model(rnn_model_path, device)
    
    print("\n=== Evaluating RNN Model ===")
    check_memory()
    rnn_loss, rnn_mae, rnn_rmse, rnn_r2, rnn_preds, rnn_actuals = evaluate_model(
        rnn_model, test_loader, nn.MSELoss(), device, scaler_y, pred_length, num_sensors
    )
    results['RNN'] = {'loss': rnn_loss, 'mae': rnn_mae, 'rmse': rnn_rmse, 'r2': rnn_r2}
    plot_predictions(rnn_preds, rnn_actuals, 'RNN')
    print(f"RNN Results - Loss: {rnn_loss:.4f}, MAE: {rnn_mae:.4f}, RMSE: {rnn_rmse:.4f}, R²: {rnn_r2:.4f}")
    
    # 4. Train and evaluate LSTM model
    if not load_saved_models or not os.path.exists('models/lstm_model.pth'):
        print("\n=== Training LSTM Model ===")
        input_dim = X_train.shape[2]
        hidden_dim = 32
        output_dim = y_train.shape[1]
        num_layers = 1
        
        lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        
        train_losses, val_losses, lstm_model = train_model(
            lstm_model, train_loader, val_loader, criterion, optimizer
        )
        lstm_model_path = save_model(lstm_model, 'lstm_model')
        plot_training_history(train_losses, val_losses, 'LSTM')
    else:
        print("\n=== Loading saved LSTM Model ===")
        lstm_model_path = 'models/lstm_model.pth'
        lstm_model = load_model(lstm_model_path, device)
    
    print("\n=== Evaluating LSTM Model ===")
    check_memory()
    lstm_loss, lstm_mae, lstm_rmse, lstm_r2, lstm_preds, lstm_actuals = evaluate_model(
        lstm_model, test_loader, nn.MSELoss(), device, scaler_y, pred_length, num_sensors
    )
    results['LSTM'] = {'loss': lstm_loss, 'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2}
    plot_predictions(lstm_preds, lstm_actuals, 'LSTM')
    print(f"LSTM Results - Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")
    
    # 5. Train and evaluate GRU model
    if not load_saved_models or not os.path.exists('models/gru_model.pth'):
        print("\n=== Training GRU Model ===")
        input_dim = X_train.shape[2]
        hidden_dim = 32
        output_dim = y_train.shape[1]
        num_layers = 1
        
        gru_model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
        
        train_losses, val_losses, gru_model = train_model(
            gru_model, train_loader, val_loader, criterion, optimizer
        )
        gru_model_path = save_model(gru_model, 'gru_model')
        plot_training_history(train_losses, val_losses, 'GRU')
    else:
        print("\n=== Loading saved GRU Model ===")
        gru_model_path = 'models/gru_model.pth'
        gru_model = load_model(gru_model_path, device)
    
    print("\n=== Evaluating GRU Model ===")
    check_memory()
    gru_loss, gru_mae, gru_rmse, gru_r2, gru_preds, gru_actuals = evaluate_model(
        gru_model, test_loader, nn.MSELoss(), device, scaler_y, pred_length, num_sensors
    )
    results['GRU'] = {'loss': gru_loss, 'mae': gru_mae, 'rmse': gru_rmse, 'r2': gru_r2}
    plot_predictions(gru_preds, gru_actuals, 'GRU')
    print(f"GRU Results - Loss: {gru_loss:.4f}, MAE: {gru_mae:.4f}, RMSE: {gru_rmse:.4f}, R²: {gru_r2:.4f}")
    
    # 6. Train and evaluate Transformer model
    if not load_saved_models or not os.path.exists('models/transformer_model.pth'):
        print("\n=== Training Transformer Model ===")
        input_dim = X_train.shape[2]
        d_model = 32
        nhead = 4
        num_layers = 1
        output_dim = y_train.shape[1]
        
        transformer_model = TransformerModel(
            input_dim, d_model, nhead, num_layers, output_dim, seq_length
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)
        
        train_losses, val_losses, transformer_model = train_model(
            transformer_model, train_loader, val_loader, criterion, optimizer
        )
        transformer_model_path = save_model(transformer_model, 'transformer_model')
        plot_training_history(train_losses, val_losses, 'Transformer')
    else:
        print("\n=== Loading saved Transformer Model ===")
        transformer_model_path = 'models/transformer_model.pth'
        transformer_model = load_model(transformer_model_path, device)
    
    print("\n=== Evaluating Transformer Model ===")
    check_memory()
    transformer_loss, transformer_mae, transformer_rmse, transformer_r2, transformer_preds, transformer_actuals = evaluate_model(
        transformer_model, test_loader, nn.MSELoss(), device, scaler_y, pred_length, num_sensors
    )
    results['Transformer'] = {'loss': transformer_loss, 'mae': transformer_mae, 'rmse': transformer_rmse, 'r2': transformer_r2}
    plot_predictions(transformer_preds, transformer_actuals, 'Transformer')
    print(f"Transformer Results - Loss: {transformer_loss:.4f}, MAE: {transformer_mae:.4f}, RMSE: {transformer_rmse:.4f}, R²: {transformer_r2:.4f}")
    
    # 7. Skip SVM to avoid crashes (optional, uncomment if needed)
    """
    if not load_saved_models or not os.path.exists('models/svm_model.pkl'):
        print("\n=== Training SVM Model ===")
        svm_model = train_svm_model(X_train, y_train)
        svm_model_path = save_model(svm_model, 'svm_model')
    else:
        print("\n=== Loading saved SVM Model ===")
        svm_model_path = 'models/svm_model.pkl'
        svm_model = load_model(svm_model_path)
    
    print("\n=== Evaluating SVM Model ===")
    check_memory()
    svm_mae, svm_rmse, svm_r2, svm_preds, svm_actuals = evaluate_svm_model(
        svm_model, X_test, y_test, scaler_y
    )
    results['SVM'] = {'mae': svm_mae, 'rmse': svm_rmse, 'r2': svm_r2}
    plot_predictions(svm_preds.reshape(-1, pred_length, num_sensors), svm_actuals.reshape(-1, pred_length, num_sensors), 'SVM')
    print(f"SVM Results - MAE: {svm_mae:.4f}, RMSE: {svm_rmse:.4f}, R²: {svm_r2:.4f}")
    """
    
    # 8. Train and evaluate Logistic Regression model
    if not load_saved_models or not os.path.exists('models/logistic_model.pkl'):
        print("\n=== Training Logistic Regression Model ===")
        threshold = np.median(y_train)
        logistic_model, threshold = train_logistic_model(X_train, y_train, threshold)
        logistic_model_path = save_model(logistic_model, 'logistic_model')
        np.save(os.path.join('models', 'logistic_threshold.npy'), threshold)
    else:
        print("\n=== Loading saved Logistic Regression Model ===")
        logistic_model_path = 'models/logistic_model.pkl'
        logistic_model = load_model(logistic_model_path)
        threshold = np.load(os.path.join('models', 'logistic_threshold.npy'))
    
    print("\n=== Evaluating Logistic Regression Model ===")
    check_memory()
    logistic_accuracy, logistic_auc, logistic_preds, logistic_actuals = evaluate_logistic_model(
        logistic_model, X_test, y_test, threshold
    )
    results['Logistic Regression'] = {'accuracy': logistic_accuracy, 'auc': logistic_auc}
    print(f"Logistic Regression Results - Accuracy: {logistic_accuracy:.4f}, AUC: {logistic_auc:.4f}")
    
    # Visualize Logistic Regression results
    plot_logistic_regression(logistic_preds, logistic_actuals, threshold, n_samples=5)
    
    # 9. Compare all models
    print("\n=== Model Comparison ===")
    check_memory()
    plot_comparison(results)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        for metric, value in model_results.items():
            print(f"  {metric}: {value:.4f}")
    
    models_with_rmse = {model: results[model]['rmse'] for model in results if 'rmse' in results.get(model, {})}
    if models_with_rmse:
        best_model = min(models_with_rmse.items(), key=lambda x: x[1])[0]
        print(f"\nBest model based on RMSE: {best_model} with RMSE = {models_with_rmse[best_model]:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file (.npz)')
    parser.add_argument('--type', type=str, default='metr_la', choices=['metr_la', 'pems_bay'], help='Dataset type')
    parser.add_argument('--load', action='store_true', help='Load saved models instead of training')
    args = parser.parse_args()
    
    results = main(args.dataset, args.type, args.load)
    
    import json
    with open('results.json', 'w') as f:
        results_serializable = {model: {metric: float(value) for metric, value in model_results.items()} 
                               for model, model_results in results.items()}
        json.dump(results_serializable, f, indent=4)
    print("Results saved to results.json")

    # Ensure plots are displayed in the terminal or environment
    plt.show()