"""
Reproduces: Table 7 (Independent external validation)

Holds out a fraction of riders entirely before model fitting, calibrates
on the remaining riders, then evaluates without retraining on the held-out
cohort. Requires a 'Cohort' column (Development / External Validation) in
Rider_Trip_Metadata.csv, assigned once at enrollment.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]

def main(stress_path, trip_path):
    stress = pd.read_csv(stress_path)
    if "Cohort" in stress.columns:
        df = stress
    else:
        trip = pd.read_csv(trip_path)
        df = stress.merge(trip[["Trip_ID", "Cohort"]], on="Trip_ID")

    dev = df[df.Cohort == "Development"]
    ext = df[df.Cohort == "External Validation"]

    X_dev, y_dev = dev[FACTORS].values, dev["Self_Reported_Stress"].values
    X_ext, y_ext = ext[FACTORS].values, ext["Self_Reported_Stress"].values

    model = LinearRegression().fit(X_dev, y_dev)

    pred_dev = model.predict(X_dev)
    pred_ext = model.predict(X_ext)

    mae_dev = mean_absolute_error(y_dev, pred_dev)
    r_dev, p_dev = stats.pearsonr(pred_dev, y_dev)

    mae_ext = mean_absolute_error(y_ext, pred_ext)
    rmse_ext = mean_squared_error(y_ext, pred_ext) ** 0.5
    r_ext, p_ext = stats.pearsonr(pred_ext, y_ext)
    rho_ext, prho_ext = stats.spearmanr(pred_ext, y_ext)

    print(f"Development cohort:  n={len(dev)}, MAE={mae_dev:.3f}, r={r_dev:.3f} (p={p_dev:.4g})")
    print(f"External Validation: n={len(ext)}, MAE={mae_ext:.3f}, RMSE={rmse_ext:.3f}, "
          f"r={r_ext:.3f} (p={p_ext:.4g}), rho={rho_ext:.3f} (p={prho_ext:.4g})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv", help="Stress_GroundTruth.csv")
    parser.add_argument("--trip-data", default="data/sample_data.csv", help="Rider_Trip_Metadata.csv")
    args = parser.parse_args()
    main(args.data, args.trip_data)
