"""
Reproduces: Table 5 (Internal validation) and Table 6 (Per-fold breakdown)

Runs 5-fold cross-validation of a linear regression over the six MFTSI
factors, predicting self-reported stress, reporting MAE and RMSE per fold.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def main(data_path, n_splits=5, seed=42):
    df = pd.read_csv(data_path)
    X = df[FACTORS].values
    y = df["Self_Reported_Stress"].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_mae, fold_rmse = [], []
    for i, (tr, te) in enumerate(kf.split(X)):
        model = LinearRegression().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        mae = mean_absolute_error(y[te], pred)
        rmse = mean_squared_error(y[te], pred) ** 0.5
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        print(f"Fold {i+1}: n_test={len(te)}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    print(f"\nMean MAE  = {np.mean(fold_mae):.3f} (SD {np.std(fold_mae):.3f})")
    print(f"Mean RMSE = {np.mean(fold_rmse):.3f} (SD {np.std(fold_rmse):.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
