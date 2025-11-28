import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class ShorelineNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ShorelineNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train,
                hidden_dim=64,
                lr=0.001,
                epochs=200,
                batch_size=64,
                device='cpu'):
    """
    Train a neural network using PyTorch.
    """

    # Convert to tensors
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ShorelineNet(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, X_test, y_test, device='cpu'):
    """
    Evaluate the neural network model.
    """
    model.eval()
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")

    return mse, rmse, r2

def save_model(model, path):
    """
    Save the neural network model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

def load_model(path, input_dim, hidden_dim=64, device='cpu'):
    """
    Load a saved neural network model.
    """
    model = ShorelineNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from: {path}")
    return model
