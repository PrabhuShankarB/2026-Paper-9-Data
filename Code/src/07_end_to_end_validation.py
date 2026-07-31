"""
Reproduces: Table 12 (RT-SFMM cause-attribution agreement)

Independently recomputes the dominant factor per segment using calibrated
weights, and compares it against RT-SFMM's recorded Dominant_Cause label,
closing the loop between MFTSI's factor decomposition and RT-SFMM's
downstream cause attribution.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FACTORS = ["Road_Damage_F1", "Wrong_Way_F2", "Illegal_Parking_F3",
           "Lane_Indiscipline_F4", "Signal_Violation_F5", "Congestion_F6"]
FACTOR_LABELS = ["F1_Road_Damage", "F2_Wrong_Way", "F3_Illegal_Parking",
                 "F4_Lane_Indiscipline", "F5_Signal_Violation", "F6_Congestion"]

def main(data_path):
    df = pd.read_csv(data_path)
    X = df[FACTORS].values
    y = df["Self_Reported_Stress"].values
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

    model = LinearRegression().fit(Xn, y)
    coefs = np.clip(model.coef_, 0, None)
    w = coefs / coefs.sum()

    weighted_contrib = Xn * w
    predicted_dominant_idx = np.argmax(weighted_contrib, axis=1)
    predicted_dominant = [FACTOR_LABELS[i] for i in predicted_dominant_idx]

    df["Predicted_Dominant_Cause"] = predicted_dominant
    overall_agreement = (df["Predicted_Dominant_Cause"] == df["Dominant_Cause"]).mean()

    print(f"Overall agreement: {overall_agreement*100:.1f}%\n")
    print("Per-factor agreement:")
    for f in FACTOR_LABELS:
        subset = df[df.Dominant_Cause == f]
        if len(subset) > 0:
            match = (subset["Predicted_Dominant_Cause"] == f).mean()
            print(f"{f:<25} n={len(subset):<5} agreement={match*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
