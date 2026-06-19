import copy
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



train_file = "5000X_trainval_with_label.csv"
test_file = "5000X_test_with_label.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train_all = train_data.iloc[:, 1:-1].values
y_train_all = train_data["PHENOTYPE"].values

X_test_raw = test_data.iloc[:, 1:-1].values
y_test_raw = test_data["PHENOTYPE"].values


mask_train = ~np.isnan(y_train_all)
X_train_all = X_train_all[mask_train]
y_train_all = y_train_all[mask_train]

mask_test = ~np.isnan(y_test_raw)
X_test_raw = X_test_raw[mask_test]
y_test_raw = y_test_raw[mask_test]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CNNRegressor(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        def conv_output_length(L_in, kernel_size, stride=1, padding=0, dilation=1):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        L1 = conv_output_length(input_length, 5)
        L2 = conv_output_length(L1, 3)
        flatten_size = 32 * L2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)



def train_one_model(X_tr, y_tr, X_val, y_val, seed,
                    batch_size=32, epochs=100, patience=10,
                    lr=1e-4):

    # Ensure reproducible model initialization and DataLoader shuffling for each fold/run.
    set_seed(seed)

    imputer = SimpleImputer(strategy='mean')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_val_imp = imputer.transform(X_val)

    scaler_X = StandardScaler()
    X_tr_scaled = scaler_X.fit_transform(X_tr_imp)
    X_val_scaled = scaler_X.transform(X_val_imp)

    scaler_y = StandardScaler()
    y_tr_scaled = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
    y_tr_tensor = torch.tensor(y_tr_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNRegressor(X_tr_tensor.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        mean_val_loss = np.mean(val_losses)

        # Dynamically reduce the learning rate when validation loss plateaus.
        scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    return model, imputer, scaler_X, scaler_y


def predict_r_p(model, X_data, y_data, imputer, scaler_X, scaler_y, batch_size=32):
    X_imp = imputer.transform(X_data)
    X_scaled = scaler_X.transform(X_imp)
    y_scaled = scaler_y.transform(y_data.reshape(-1, 1)).flatten()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_orig = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    r, p = pearsonr(targets_orig, preds_orig)
    return r, p


n_repeats = 100
n_splits = 10

all_cv_results = []
cv_mean_r_list = []

# Repeated 10-fold cross-validation on the training/validation set.
for run in range(1, n_repeats + 1):
    seed = run
    set_seed(seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_r_list = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_all), start=1):
        X_tr, X_val = X_train_all[tr_idx], X_train_all[val_idx]
        y_tr, y_val = y_train_all[tr_idx], y_train_all[val_idx]

        model, imputer, scaler_X, scaler_y = train_one_model(
            X_tr, y_tr, X_val, y_val, seed=seed * 100 + fold
        )

        fold_r, _ = predict_r_p(model, X_val, y_val, imputer, scaler_X, scaler_y)
        fold_r_list.append(fold_r)

    mean_cv_r = np.mean(fold_r_list)
    cv_mean_r_list.append(mean_cv_r)

    all_cv_results.append({
        "Run": run,
        "Seed": seed,
        "CV_Mean_r": mean_cv_r
    })

    print(f"Run {run}: CV mean r = {mean_cv_r:.4f}")


# Train the final model once, then evaluate it once on the independent test set.
final_seed = 42
set_seed(final_seed)

X_subtrain, X_internal_val, y_subtrain, y_internal_val = train_test_split(
    X_train_all, y_train_all, test_size=0.1, random_state=final_seed
)

final_model, final_imputer, final_scaler_X, final_scaler_y = train_one_model(
    X_subtrain, y_subtrain, X_internal_val, y_internal_val, seed=final_seed + 999
)

test_r, test_p = predict_r_p(
    final_model, X_test_raw, y_test_raw,
    final_imputer, final_scaler_X, final_scaler_y
)

print(f"Independent test: R = {test_r:.4f}, P = {test_p:.4e}")


cv_results_df = pd.DataFrame(all_cv_results)
cv_results_df.to_excel("BrDeepGS_repeatedCV_results.xlsx", index=False)

summary_df = pd.DataFrame({
    "Metric": [
        "Mean_CV_r",
        "SD_CV_r",
        "Independent_Test_r",
        "Independent_Test_p"
    ],
    "Value": [
        cv_results_df["CV_Mean_r"].mean(),
        cv_results_df["CV_Mean_r"].std(ddof=1),
        test_r,
        test_p
    ]
})
summary_df.to_excel("BrDeepGS_repeatedCV_independentTest_summary.xlsx", index=False)

print("Done.")
