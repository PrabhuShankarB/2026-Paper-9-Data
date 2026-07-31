# MFTSI + SVFASP + RT-SFMM: Traffic-Induced Rider Stress Method

Reproducible implementation for "A Sensor-Vision Fusion Method for Real-Time Detection and Mitigation of Traffic-Induced Rider Stress."

## What this repository contains

Nine scripts, each reproducing one validation result reported in the manuscript. Each script is self-contained, documented, and runs against the four CSV files described in `data/schema.md`.

| Script | Reproduces | Manuscript reference |
|---|---|---|
| `01_ground_truth_correlation.py` | Pearson/Spearman correlation between MFTSI and self-reported stress | Table 4 |
| `02_internal_validation.py` | 5-fold cross-validation, MAE/RMSE | Table 5, Table 6 |
| `03_external_validation.py` | Rider-holdout generalization test | Table 7 |
| `04_ablation.py` | Per-factor marginal contribution | Table 8 |
| `05_uncertainty_bootstrap.py` | Bootstrap 95% confidence intervals | Table 9, Table 10 |
| `06_calibration_stability.py` | Bootstrap stability of calibrated weights | Table 11 |
| `07_end_to_end_validation.py` | RT-SFMM cause-attribution agreement check | Table 12 |
| `08_sensitivity_analysis.py` | Simulated dropout/noise degradation | Table 17 |
| `09_intervention_efficacy.py` | Pre/post stress comparison by feedback condition | Table 18, Table 19 |

## Data availability

The real primary dataset (25 riders, 250 trips, 1,000 segments) includes physiological data (blood pressure, heart rate) and is not published here, consistent with participant privacy and the institutional ethics review referenced in the manuscript. `data/schema.md` documents the exact column structure. `data/sample_data.csv` contains a small, clearly labeled synthetic sample matching this schema, provided only so the scripts below can be run and their logic inspected — it is not real data and should not be used to draw any conclusion about the method's actual performance.

## Setup

```bash
pip install -r requirements.txt
```

## Running the analyses

Each script accepts a `--data` argument pointing to your own CSV files matching the schema, or defaults to `data/sample_data.csv` for a structural test run:

```bash
python src/01_ground_truth_correlation.py --data path/to/your/Stress_GroundTruth.csv
```

Running against the synthetic sample data will execute successfully but will not reproduce the manuscript's actual numbers — only real data collected under the described protocol will do that.

## Method summary

- **MFTSI** (Multi-Factor Traffic Stress Index): a composite score from six normalized, regression-calibrated road-condition factors (road damage, wrong-way vehicles, illegal parking, lane indiscipline, signal violation, congestion).
- **SVFASP** (Sensor-Vision Fused Adaptive Stress Prediction): a model fusing IMU and dashcam-derived features to predict stress in real time (validated here via a linear-regression proxy given current data volume; a GRU is the intended production model, see manuscript Section 5.3).
- **RT-SFMM** (Real-Time Stress Feedback and Mitigation Module): attributes predicted stress to its dominant cause and issues a post-trip, cause-specific recommendation.

## License

MIT License — see `LICENSE`.

## Citation

If you use this method or code, please cite the associated MethodsX manuscript (details to be added upon publication).
