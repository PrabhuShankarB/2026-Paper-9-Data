"""
Reproduces: Table 17 (Sensitivity to simulated data degradation)

Simulates missing sensor readings (dropout) and sensor noise at increasing
severity, re-evaluating 5-fold CV MAE at each level, averaged over multiple
trials per severity.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def eval_cv(X, y, seed=42):
    maes = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        model = LinearRegression().fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[te], model.predict(X[te])))
    return np.mean(maes)

def main(data_path, n_trials=20):
    df = pd.read_csv(data_path)
    X = df[FACTORS].values.astype(float)
    y = df["Self_Reported_Stress"].values

    clean_mae = eval_cv(X, y)
    print(f"Clean baseline MAE: {clean_mae:.3f}\n")

    print("Missing data (dropout):")
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        trial_maes = []
        for trial in range(n_trials):
            Xd = X.copy()
            mask = np.random.RandomState(trial).rand(*Xd.shape) < pct
            Xd[mask] = 0
            trial_maes.append(eval_cv(Xd, y, seed=trial))
        print(f"  {pct*100:.0f}%: MAE={np.mean(trial_maes):.3f}, "
              f"degradation={np.mean(trial_maes)-clean_mae:+.3f}")

    print("\nSensor noise (Gaussian, % of feature range):")
    feature_ranges = X.max(axis=0) - X.min(axis=0)
    for noise_pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
        trial_maes = []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            noise = rng.normal(0, noise_pct * feature_ranges, size=X.shape)
            trial_maes.append(eval_cv(X + noise, y, seed=trial))
        print(f"  {noise_pct*100:.0f}%: MAE={np.mean(trial_maes):.3f}, "
              f"degradation={np.mean(trial_maes)-clean_mae:+.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
