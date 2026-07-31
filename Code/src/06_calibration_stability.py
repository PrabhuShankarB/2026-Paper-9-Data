"""
Reproduces: Table 11 (Bootstrap stability of calibrated factor weights)

Re-estimates the regression-calibrated MFTSI weights across many bootstrap
resamples to check whether calibration is stable or sample-dependent.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def main(data_path, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.read_csv(data_path)
    X = df[FACTORS].values
    y = df["Self_Reported_Stress"].values
    n = len(df)

    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

    boot_weights = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        model = LinearRegression().fit(Xn[idx], y[idx])
        coefs = np.clip(model.coef_, 0, None)
        w = coefs / coefs.sum() if coefs.sum() > 0 else np.ones(len(FACTORS)) / len(FACTORS)
        boot_weights.append(w)
    boot_weights = np.array(boot_weights)

    print(f"{'Factor':<25}{'Mean weight':>12}{'SD':>10}{'95% CI':>22}{'CoV':>8}")
    for i, factor in enumerate(FACTORS):
        w = boot_weights[:, i]
        ci = np.percentile(w, [2.5, 97.5])
        cov = w.std() / w.mean() if w.mean() > 0 else float("nan")
        print(f"{factor:<25}{w.mean():>12.3f}{w.std():>10.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]{cov:>8.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
