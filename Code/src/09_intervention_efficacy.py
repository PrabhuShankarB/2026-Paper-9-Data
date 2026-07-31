"""
Reproduces: Table 18 (Intervention efficacy, overall) and
            Table 19 (Intervention efficacy by road type)

Compares pre/post stress depending on whether the rider followed the
RT-SFMM recommendation, overall and broken down by road type.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def main(stress_path, trip_path=None):
    df = pd.read_csv(stress_path)
    if trip_path:
        trip = pd.read_csv(trip_path)
        df = df.merge(trip[["Trip_ID", "Road_Type"]], on="Trip_ID")

    followed = df[df.Recommendation_Followed == "Yes"]
    not_followed = df[df.Recommendation_Followed == "No"]

    print("Overall:")
    print(f"  Followed (n={len(followed)}): "
          f"{followed.Self_Reported_Stress.mean():.3f} -> {followed.Post_Trip_Stress.mean():.3f}")
    print(f"  Not followed (n={len(not_followed)}): "
          f"{not_followed.Self_Reported_Stress.mean():.3f} -> {not_followed.Post_Trip_Stress.mean():.3f}")

    delta_f = followed.Post_Trip_Stress - followed.Self_Reported_Stress
    delta_n = not_followed.Post_Trip_Stress - not_followed.Self_Reported_Stress
    t, p = stats.ttest_ind(delta_f, delta_n)
    pooled_sd = np.sqrt(((len(delta_f)-1)*delta_f.var() + (len(delta_n)-1)*delta_n.var()) /
                         (len(delta_f)+len(delta_n)-2))
    cohens_d = (delta_f.mean() - delta_n.mean()) / pooled_sd
    print(f"\n  Independent t-test: t={t:.3f}, p={p:.4g}, Cohen's d={cohens_d:.3f}")

    if "Road_Type" in df.columns:
        print("\nBy road type:")
        for rt, g in df.groupby("Road_Type"):
            gf = g[g.Recommendation_Followed == "Yes"]
            gn = g[g.Recommendation_Followed == "No"]
            if len(gf) > 3 and len(gn) > 3:
                d_f = (gf.Post_Trip_Stress - gf.Self_Reported_Stress).mean()
                d_n = (gn.Post_Trip_Stress - gn.Self_Reported_Stress).mean()
                print(f"  {rt:<22} followed(n={len(gf)}) delta={d_f:.3f} | "
                      f"not-followed(n={len(gn)}) delta={d_n:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv", help="Stress_GroundTruth.csv")
    parser.add_argument("--trip-data", default=None, help="Rider_Trip_Metadata.csv (optional, for road-type breakdown)")
    args = parser.parse_args()
    main(args.data, args.trip_data)
