"""
Reproduces: Table 8 (Ablation results)

Retrains the model with each factor removed in turn, via 5-fold CV,
reporting the increase in MAE and RMSE relative to the full model.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def cv_scores(X, y, seed=42, n_splits=5):
    maes, rmses = [], []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        model = LinearRegression().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(mean_squared_error(y[te], pred) ** 0.5)
    return np.mean(maes), np.mean(rmses)

def main(data_path):
    df = pd.read_csv(data_path)
    X = df[FACTORS].values
    y = df["Self_Reported_Stress"].values

    full_mae, full_rmse = cv_scores(X, y)
    print(f"Full model: MAE={full_mae:.3f}, RMSE={full_rmse:.3f}\n")

    results = []
    for i, factor in enumerate(FACTORS):
        idx = [j for j in range(len(FACTORS)) if j != i]
        mae, rmse = cv_scores(X[:, idx], y)
        results.append((factor, mae - full_mae, rmse - full_rmse))

    results.sort(key=lambda r: -r[1])
    for factor, d_mae, d_rmse in results:
        print(f"Without {factor:<25}: delta MAE={d_mae:+.3f}, delta RMSE={d_rmse:+.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
