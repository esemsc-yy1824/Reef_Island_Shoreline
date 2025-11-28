import optuna
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models.neural_net import ShorelineNet

def objective(trial, X_train_full, y_train_full, device='cpu'):
    # Split training data into a smaller train/validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42
    )

    # Suggest hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    epochs = trial.suggest_int('epochs', 100, 500)

    # Prepare tensors
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    # y_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    val_X_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    # val_y_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(device)
    val_y_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Build model
    model = ShorelineNet(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor).cpu().numpy().flatten()

    mse = mean_squared_error(y_val, val_preds)
    return mse

def run_optuna(X_train_full, y_train_full, n_trials=30, device='cpu'):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train_full, y_train_full, device=device),
        n_trials=n_trials
    )

    print("Best trial:")
    print(study.best_trial.params)
    print("Best MSE:", study.best_value)

    return study.best_trial.params
