# Data schema

The method expects four CSV files, joined on `Rider_ID` and `Trip_ID`.

## Rider_Trip_Metadata.csv

| Column | Type | Description |
|---|---|---|
| Rider_ID | string | Unique rider identifier (e.g. R001) |
| Trip_ID | string | Unique trip identifier |
| Rider_Age | int | Rider age in years |
| Riding_Experience_Years | int | Years of riding experience (must satisfy Age - Experience >= legal minimum riding age) |
| Route_ID | string | Route identifier, prefixed by city (e.g. R_CHE_01) |
| Road_Type | string | Road-type category (e.g. Commercial Street, Highway/ORR) |
| Cohort | string | "Development" or "External Validation" — assigned once at enrollment, never changed |

## Sensor_Data.csv

| Column | Type | Description |
|---|---|---|
| Rider_ID, Trip_ID, Segment_ID | string | Join keys |
| Timestamp | datetime | Sample timestamp |
| Latitude, Longitude | float | GPS coordinates |
| Speed_kmh | float | Instantaneous speed |
| Accel_X, Accel_Y, Accel_Z | float | Tri-axial accelerometer readings (m/s^2) |
| Gyro_X, Gyro_Y, Gyro_Z | float | Tri-axial gyroscope readings |
| Accel_RMS | float | RMS acceleration magnitude, gravity-removed (F1 input) |
| Crest_Factor | float | Peak-to-RMS ratio (F1 input) |
| Stop_Ratio | float | Fraction of segment time below 5 km/h (F6 input) |

## Vision_Event_Data.csv

| Column | Type | Description |
|---|---|---|
| Rider_ID, Trip_ID, Segment_ID | string | Join keys |
| Wrong_Way_Count | int | Wrong-way vehicles detected (F2) |
| Illegal_Parking_Pct | float | Percent of lane width occluded by parking (F3) |
| Lane_Indiscipline_Count | int | Lane-cutting/parallel-riding events (F4) |
| Signal_Violation_Count | int | Red-signal or blocked-green events (F5) |

## Stress_GroundTruth.csv

| Column | Type | Description |
|---|---|---|
| Rider_ID, Trip_ID, Segment_ID | string | Join keys |
| Road_Damage_F1, Wrong_Way_F2, Illegal_Parking_F3, Lane_Indiscipline_F4, Signal_Violation_F5, Congestion_F6 | float | Normalized [0,1] MFTSI factor values |
| MFTSI_Score | float | Composite index (0-100 scale in this dataset) |
| Self_Reported_Stress | int | Rider-reported stress, 0-10 |
| Predicted_Stress | float | SVFASP model output |
| Dominant_Cause | string | RT-SFMM's attributed cause label |
| RT_SFMM_Recommendation | string | Recommendation text/category issued |
| Recommendation_Followed | string | "Yes" / "No" |
| Post_Trip_Stress | int | Stress reported after the recommendation window |
