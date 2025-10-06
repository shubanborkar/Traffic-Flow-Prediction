# inference.py
import numpy as np
import torch
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import functions from main script
from main import load_model, load_metr_la_data, load_pems_bay_data, preprocess_data

def predict_future_traffic(model_path, data_path, dataset_type, nodes_to_visualize=3, horizon=12):
    """
    Make predictions for future traffic using a trained model
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        dataset_type: Type of dataset ('metr_la' or 'pems_bay')
        nodes_to_visualize: Number of sensor nodes to visualize
        horizon: Number of time steps to predict into the future
    """
    # Load data
    if dataset_type == 'metr_la':
        data = load_metr_la_data(data_path)
    elif dataset_type == 'pems_bay':
        data = load_pems_bay_data(data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Load model and scaler
    model = load_model(model_path)
    scaler_path = model_path.replace('.pth', '_scaler.pkl').replace('.pkl', '_scaler.pkl')
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"Scaler not found at {scaler_path}, using StandardScaler")
        scaler = StandardScaler()
        # Fit the scaler on the data
        data_flat = data.reshape(-1, data.shape[-1])
        scaler.fit(data_flat)
    
    # Get the latest data for prediction
    seq_length = 12  # Same as used in training
    latest_data = data[-seq_length:]
    
    # Preprocess the latest data
    X_latest, _ = preprocess_data(latest_data, seq_length, 1)
    
    # Normalize the data
    X_latest_flat = X_latest.reshape(-1, X_latest.shape[-1])
    X_latest_norm = scaler.transform(X_latest_flat).reshape(X_latest.shape)
    
    # Convert to PyTorch tensor
    X_latest_tensor = torch.FloatTensor(X_latest_norm).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make predictions for multiple time steps
    predictions = []
    current_input = X_latest_tensor
    
    # Set model to evaluation mode
    if isinstance(model, torch.nn.Module):
        model.eval()
    
    with torch.no_grad():
        for _ in range(horizon):
            # Make prediction
            if isinstance(model, torch.nn.Module):
                pred = model(current_input)
                
                # For PyTorch models, reshape prediction to match the input shape for next step
                pred_reshaped = pred.view(1, 1, -1)
                
                # Update input by removing the first time step and adding the prediction
                current_input = torch.cat([current_input[:, 1:, :], pred_reshaped], dim=1)
            else:
                # For scikit-learn models
                current_input_np = current_input.cpu().numpy()
                current_input_flat = current_input_np.reshape(current_input_np.shape[0], -1)
                pred = model.predict(current_input_flat)
                
                # Reshape prediction and update input
                pred_reshaped = pred.reshape(1, 1, -1)
                current_input_np = np.concatenate([current_input_np[:, 1:, :], pred_reshaped], axis=1)
                current_input = torch.FloatTensor(current_input_np).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Store prediction
            predictions.append(pred.cpu().numpy())
    
    # Convert predictions to numpy array
    predictions = np.array(predictions).squeeze()
    
    # Inverse transform predictions
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    else:
        predictions = predictions.reshape(predictions.shape[0], -1)
    
    predictions_inv = scaler.inverse_transform(predictions)
    
    # Reshape to original format if needed
    if dataset_type == 'metr_la':
        n_nodes = 207  # METR-LA has 207 sensors
    elif dataset_type == 'pems_bay':
        n_nodes = 325  # PEMS-BAY has 325 sensors
    else:
        n_nodes = int(np.sqrt(predictions_inv.shape[1]))
    
    predictions_inv = predictions_inv.reshape(horizon, n_nodes, -1)
    
    # Plot predictions for selected nodes
    plt.figure(figsize=(14, 8))
    
    for i in range(nodes_to_visualize):
        plt.subplot(nodes_to_visualize, 1, i+1)
        plt.plot(predictions_inv[:, i, 0], 'b-o', label=f'Node {i+1} Prediction')
        plt.title(f'Traffic Flow Prediction for Node {i+1}')
        plt.xlabel('Time Steps into Future')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('future_predictions.png')
    
    print(f"Made predictions for {horizon} time steps into the future")
    print(f"Predictions visualization saved to future_predictions.png")
    
    return predictions_inv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file (.npz)')
    parser.add_argument('--type', type=str, default='metr_la', choices=['metr_la', 'pems_bay'], help='Dataset type')
    parser.add_argument('--nodes', type=int, default=3, help='Number of nodes to visualize')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon')
    args = parser.parse_args()
    
    predictions = predict_future_traffic(args.model, args.dataset, args.type, args.nodes, args.horizon)
    
    # Save predictions to file
    np.save('future_predictions.npy', predictions)
    print("Predictions saved to future_predictions.npy")