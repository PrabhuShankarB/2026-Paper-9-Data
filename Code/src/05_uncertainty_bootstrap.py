"""
Reproduces: Table 9 (Bootstrap CIs for key estimates) and
            Table 10 (Bootstrap CIs for ablation deltas)

Bootstrap resampling to compute 95% confidence intervals via the
percentile method.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def main(data_path, n_boot=2000, n_boot_ablation=500, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.read_csv(data_path)
    X = df[FACTORS].values
    y = df["Self_Reported_Stress"].values
    mftsi = df["MFTSI_Score"].values
    n = len(df)

    # --- Ground-truth correlation CI ---
    boot_r = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = stats.pearsonr(mftsi[idx], y[idx])
        boot_r.append(r)
    ci_r = np.percentile(boot_r, [2.5, 97.5])
    print(f"Ground-truth correlation: 95% CI = [{ci_r[0]:.3f}, {ci_r[1]:.3f}]")

    # --- Internal validation MAE CI ---
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(n)
    for tr, te in kf.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    resid = np.abs(preds - y)
    boot_mae = [resid[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)]
    ci_mae = np.percentile(boot_mae, [2.5, 97.5])
    print(f"Internal validation MAE: 95% CI = [{ci_mae[0]:.3f}, {ci_mae[1]:.3f}]")

    # --- Ablation delta CIs (Table 10) ---
    print("\nAblation delta 95% CIs (n_boot={}):".format(n_boot_ablation))
    for i, factor in enumerate(FACTORS):
        deltas = []
        for b in range(n_boot_ablation):
            idx = rng.choice(n, n, replace=True)
            Xb, yb = X[idx], y[idx]
            kf_b = KFold(n_splits=5, shuffle=True, random_state=b)
            full_maes = [mean_absolute_error(yb[te], LinearRegression().fit(Xb[tr], yb[tr]).predict(Xb[te]))
                         for tr, te in kf_b.split(Xb)]
            sub_idx = [j for j in range(len(FACTORS)) if j != i]
            Xsub = Xb[:, sub_idx]
            sub_maes = [mean_absolute_error(yb[te], LinearRegression().fit(Xsub[tr], yb[tr]).predict(Xsub[te]))
                        for tr, te in KFold(n_splits=5, shuffle=True, random_state=b).split(Xsub)]
            deltas.append(np.mean(sub_maes) - np.mean(full_maes))
        ci = np.percentile(deltas, [2.5, 97.5])
        print(f"{factor:<25} delta={np.mean(deltas):+.3f}, 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
