"""
Reproduces: Table 4 (Ground-truth correlation, before vs. after)

Computes Pearson and Spearman correlation between the MFTSI composite score
and self-reported stress across all segments.
"""
import argparse
import pandas as pd
from scipy import stats

def main(data_path):
    df = pd.read_csv(data_path)
    r, p = stats.pearsonr(df["MFTSI_Score"], df["Self_Reported_Stress"])
    rho, p_rho = stats.spearmanr(df["MFTSI_Score"], df["Self_Reported_Stress"])

    print(f"n = {len(df)}")
    print(f"Pearson r  = {r:.3f} (p = {p:.4g})")
    print(f"Spearman rho = {rho:.3f} (p = {p_rho:.4g})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv",
                         help="Path to Stress_GroundTruth.csv (or merged equivalent)")
    args = parser.parse_args()
    main(args.data)
