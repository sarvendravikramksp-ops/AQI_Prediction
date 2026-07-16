"""
Delhi AQI pipeline
1. Data cleaning / anomaly fixing
2. Feature engineering
3. AQI target construction + Random Forest model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# CONSTANTS
# ============================================================
MIN_FREEZE_RUN = 5
IQR_MULTIPLIER = 2.5
NIGHT_O3_THRESHOLD = 40
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def fix_sensor_freeze(df, col, min_run=5):
    """Replace runs of >= min_run identical consecutive values with NaN,
    then interpolate over time.

    FIX: the previous version used `value == shift(1)` to mark rows, which
    undercounts every run's length by 1 (the first value in a frozen run
    is never "same as previous") and never replaces that first value.
    This version groups consecutive equal values directly, so a run's
    full length is measured and every member of it (including the first)
    gets replaced.
    """
    values = df[col]
    group_id = (values != values.shift(1)).cumsum()
    run_length = values.groupby(group_id).transform("size")
    mask = run_length >= min_run

    n_replaced = int(mask.sum())
    df.loc[mask, col] = np.nan

    df = df.set_index("date")
    df[col] = df[col].interpolate(method="time")
    df = df.reset_index()

    print(f"  {col}: {n_replaced} frozen values replaced via interpolation")
    return df


def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def get_pm25_subindex(x):
    if pd.isna(x):
        return 0
    elif x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    else:
        return 400 + (x - 250) * 100 / 130


def get_pm10_subindex(x):
    if pd.isna(x):
        return 0
    elif x <= 50:
        return x
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250) * 100 / 100
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    else:
        return 400 + (x - 430) * 100 / 80


# ============================================================
# STEP 1 — LOAD + BASIC CHECKS
# ============================================================
df = pd.read_csv("delhi_aqi.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

print("Shape before anomaly fixes:", df.shape)

# --- Missing values ---
if df.isnull().sum().sum() == 0:
    print("\nFix-1\n\tNo missing value")
else:
    print("Missing value: ", df.isnull().sum().sum())

# --- Duplicates ---
date_dups = df.duplicated(subset="date").sum()
if df.duplicated().sum() == 0 and date_dups == 0:
    print("\nFix-2\n\tNo duplicated row")
else:
    print(f"duplicated row: {df.duplicated().sum()}")
    print(f"duplicate timestamps: {date_dups}")

# --- Negative values ---
neg_found = False
for col in numeric_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        neg_found = True
        print(f"{col}:{neg} negative values")
        print(df[df[col] < 0][["date", col]].head(5).to_string())
if not neg_found:
    print("\nFix-3\n\tNo negative values found.")


# ============================================================
# STEP 2 — FILL TIME GAPS
# ============================================================
full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="1h")
df = df.set_index("date").reindex(full_range)  # inserts NaN rows for gaps
df.index.name = "date"
df[numeric_cols] = df[numeric_cols].interpolate(method="time")
df = df.reset_index()


# ============================================================
# STEP 3 — OUTLIER CAPPING (IQR)
# ============================================================
print("\nFix 4 — Outlier capping (IQR):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR

    n_outliers = (df[col] > upper_fence).sum()
    if n_outliers > 0:
        df[col] = df[col].clip(upper=upper_fence)
        print(f"  {col}: {n_outliers} values capped at {upper_fence:.2f}")
    else:
        print(f"  {col}: no outliers above IQR fence")


# ============================================================
# STEP 4 — SENSOR FREEZE, PASS 1 (before capping)
# ============================================================
print("\nFix 5 — Sensor freeze (pass 1 — before capping):")
for col in ["no", "o3"]:
    df = fix_sensor_freeze(df, col, MIN_FREEZE_RUN)


# ============================================================
# STEP 5 — WIDER 2.5x IQR CEILING
# ============================================================
caps_25 = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    caps_25[col] = Q3 + IQR_MULTIPLIER * IQR

for col, new_cap in caps_25.items():
    old_cap_rows = (df[col] >= df[col].max() - 0.1).sum()  # rows at old ceiling
    print(
        f"  {col}: old cap={df[col].max():.2f} | new cap={new_cap:.2f} "
        f"| rows released from old ceiling={old_cap_rows}"
    )


# ============================================================
# STEP 6 — NIGHTTIME O3 CORRECTION
# ============================================================
print("\nFix 6 — Nighttime O3 correction:")
df = df.set_index("date")
df["hour"] = df.index.hour

night_mask = (
    (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
) & (df["o3"] > NIGHT_O3_THRESHOLD)

n_fixed = night_mask.sum()
df.loc[night_mask, "o3"] = np.nan
df["o3"] = df["o3"].interpolate(method="time")

night_hours = (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
still_high = (night_hours & (df["o3"] > NIGHT_O3_THRESHOLD)).sum()
df.loc[night_hours & (df["o3"] > NIGHT_O3_THRESHOLD), "o3"] = NIGHT_O3_THRESHOLD
df = df.drop(columns="hour").reset_index()

print(f"  {n_fixed} nighttime O3 rows corrected")
print(f"  {still_high} rows hard-capped at {NIGHT_O3_THRESHOLD} µg/m³ after interpolation")


# ============================================================
# STEP 7 — SMOOTH CAP PLATEAUS + APPLY 2.5x CEILING
# ============================================================
print("\nFix 7 — Smooth cap plateaus + apply 2.5x ceiling:")
df = df.set_index("date")

for col in numeric_cols:
    old_ceiling = df[col].max()
    new_ceiling = caps_25[col]
    at_ceiling = df[col] >= old_ceiling - 0.01
    run_id = (at_ceiling != at_ceiling.shift()).cumsum()
    plateau_count = 0

    for gid, group in df[at_ceiling].groupby(run_id[at_ceiling]):
        if len(group) >= 2:
            interior_idx = group.index[1:-1]
            if len(interior_idx) > 0:
                df.loc[interior_idx, col] = np.nan
                plateau_count += 1

    df[col] = df[col].interpolate(method="time")
    df[col] = df[col].clip(upper=new_ceiling)

    remaining_plateaus = (df[col] >= new_ceiling - 0.01).sum()
    print(
        f"  {col}: {plateau_count} plateaus smoothed | "
        f"new ceiling={new_ceiling:.2f} | rows at new ceiling={remaining_plateaus}"
    )

# --- Post-smooth freeze cleanup ---
# FIX: use the same corrected run-length logic as fix_sensor_freeze (see
# comment there) instead of the undercounting `value == shift(1)` approach.
print("\n  Post-smooth freeze cleanup:")
for col in numeric_cols:
    group_id = (df[col] != df[col].shift(1)).cumsum()
    run_length = df[col].groupby(group_id).transform("size")
    mask = run_length >= 5

    if not mask.any():
        print(f"  {col}: OK")
        continue

    n_replaced = int(mask.sum())
    df.loc[mask, col] = np.nan
    df[col] = df[col].interpolate(method="time")
    df[col] = df[col].clip(upper=caps_25[col])

    if col == "o3":
        night_mask2 = (df.index.hour >= NIGHT_START_HOUR) | (df.index.hour < NIGHT_END_HOUR)
        df.loc[night_mask2 & (df[col] > NIGHT_O3_THRESHOLD), col] = NIGHT_O3_THRESHOLD

    print(f"  {col}: {n_replaced} post-smooth frozen values re-interpolated")

df = df.reset_index()


# ============================================================
# STEP 8 — VERIFICATION
# ============================================================
print("VERIFICATION")

gaps = (df["date"].diff().dropna() > pd.Timedelta("1h")).sum()
print(f"Time gaps             : {gaps}")

# FIX: same corrected run-length logic, so this check actually reflects
# what fix_sensor_freeze / the post-smooth cleanup consider a "run".
df_tmp = df.set_index("date")
max_freezes = {}
for col in numeric_cols:
    series = df_tmp[col].copy()
    if col == "o3":
        night_hrs = (df_tmp.index.hour >= NIGHT_START_HOUR) | (df_tmp.index.hour < NIGHT_END_HOUR)
        series[night_hrs & (series == NIGHT_O3_THRESHOLD)] = np.nan
    group_id = (series != series.shift(1)).cumsum()
    run_lengths_per_row = series.groupby(group_id).transform("size")
    max_freezes[col] = int(run_lengths_per_row.max())

freeze_ok = all(v < 5 for v in max_freezes.values())
print(
    f"Max freeze runs       : {max(max_freezes.values())} "
    f"({'good all < 5' if freeze_ok else 'bad'})"
)

df["hour"] = df["date"].dt.hour
night_o3 = df[df["hour"].isin([0, 1, 2, 3])]["o3"]
print(
    f"Night O3 (0–3am) max  : {night_o3.max():.2f}  "
    f"({'good' if night_o3.max() <= NIGHT_O3_THRESHOLD else 'bad'})"
)
print(f"Night O3 (0–3am) mean : {night_o3.mean():.2f}")

print("Cliff edges remaining :")
for col in numeric_cols:
    cap = caps_25[col]
    at_cap = df[col] >= cap * 0.98
    next_val = df[col].shift(-1)
    # FIX: guard against divide-by-zero when df[col] == 0 (would otherwise
    # produce inf and get spuriously flagged as a "cliff").
    safe_denom = df[col].replace(0, np.nan)
    big_drop = (df[col] - next_val) / safe_denom > 0.40
    cliff = (at_cap & big_drop).sum()
    tag = "good" if cliff == 0 else "bad"
    print(f"  {col}: {cliff} {tag}")

print(f"pm2_5 ~ pm10 corr     : {df['pm2_5'].corr(df['pm10']):.3f}  (expect > 0.85)")
print(f"no ~ no2 corr         : {df['no'].corr(df['no2']):.3f}  (expect > 0.5)")

df = df.drop(columns="hour")
print(f"\nFinal shape           : {df.shape}")
df[numeric_cols] = df[numeric_cols].round(2)

df.to_csv("delhi_aqi_final.csv", index=False)
print("Saved → delhi_aqi_final.csv")


# ============================================================
# STEP 9 — FEATURE ENGINEERING
# ============================================================
df["date"] = pd.to_datetime(df["date"])

df["hour"] = df["date"].dt.hour
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["season"] = df["month"].apply(get_season)

# --- Cyclical encodings ---
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["season_sin"] = np.sin(2 * np.pi * df["season"] / 4)
df["season_cos"] = np.cos(2 * np.pi * df["season"] / 4)

df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# --- Lag + rolling features ---
pollutant_columns = ["pm2_5", "pm10", "no", "no2", "nh3", "co", "so2", "o3"]
pollutant_columns = [col for col in pollutant_columns if col in df.columns]

for col in pollutant_columns:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag24"] = df[col].shift(24)

# FIX: rolling windows previously included the CURRENT row (pandas'
# .rolling() is trailing-inclusive), so pm2_5_rolling3/24 and
# pm10_rolling3/24 leaked the current-row pollutant value straight into
# a feature — and the AQI target is built from that same current-row
# value. shift(1) before rolling makes the window strictly past hours.
for col in ["pm2_5", "pm10", "co"]:
    if col in df.columns:
        df[f"{col}_rolling3"] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
        df[f"{col}_rolling24"] = df[col].shift(1).rolling(window=24, min_periods=1).mean()


# Drop the first 24 rows instead of bfill() — those rows have NaN in the
# _lag24 columns (nothing 24 hours earlier exists yet), and backward-filling
# them would pull FUTURE values backward into those rows, which is a
# (small) leak. We have plenty of data, so just drop them.
df = df.iloc[24:].reset_index(drop=True)

# Keep raw (unscaled) pm2_5 / pm10 for the AQI sub-index formulas later —
# those formulas use real-world µg/m³ thresholds (e.g. <=30, <=60), so they
# must NOT be run on standardized/scaled values.
df["pm2_5_raw"] = df["pm2_5"]
df["pm10_raw"] = df["pm10"]

print("\nFeature Engineering Completed Successfully")
print("Shape after Feature Engineering:", df.shape)

print("\nNew Features Added:")
print([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season_sin", "season_cos",
    "weekday_sin", "weekday_cos",
])

print("\nFirst Five Rows:")
print(df.head())

df.to_csv("delhi_aqi_feature_engineered.csv", index=False)
print("\nSaved → delhi_aqi_feature_engineered.csv")


# ============================================================
# STEP 10 — AQI TARGET + MODEL TRAINING
# ============================================================
df = pd.read_csv("delhi_aqi_feature_engineered.csv")

if "date" in df.columns:
    df = df.drop("date", axis=1)

# AQI computed from the RAW (unscaled) pm2_5 / pm10 values.
df["pm25_aqi"] = df["pm2_5_raw"].apply(get_pm25_subindex)
df["pm10_aqi"] = df["pm10_raw"].apply(get_pm10_subindex)
df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1).round(0)
df = df.drop(columns=["pm25_aqi", "pm10_aqi", "pm2_5_raw", "pm10_raw"])

target_col = "aqi" if "aqi" in df.columns else "pm2_5"
X = df.drop(columns=[target_col, "pm2_5", "pm10"], errors="ignore")
y = df[target_col]

# Chronological split (no shuffling) — the test set is a genuine block of
# "future" rows the model has never seen, which matters because lag/rolling
# features make neighboring rows look very similar.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_train = X_train.copy()
X_test = X_test.copy()

# No feature scaling: RandomForestRegressor splits on relative order, not
# magnitude, so StandardScaler here would have zero effect on predictions
# or feature importances — just extra work.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Target Variable Used: {target_col}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# --- Feature importance plot ---
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=True).tail(15)

plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Chart successfully saved as feature_importance.png!")

# Actual vs Predicted AQI
plt.figure(figsize=(8, 8))

plt.scatter(y_test, predictions, alpha=0.6, color="steelblue")

# ============================================================
# ACTUAL vs PREDICTED AQI SCATTER PLOT
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5,
    color="royalblue",
    s=25,
    label="Predicted points"
)

# Perfect prediction line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linewidth=2,
    label="Perfect prediction (Actual = Predicted)"
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI - Random Forest")

plt.legend()
plt.grid(alpha=0.3)

# Show R2 and MAE on graph
plt.text(
    0.05,
    0.95,
    f"R² = {r2:.2f}\nMAE = {mae:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig("actual_vs_predicted_aqi.png", dpi=300)
plt.show()

print("Saved → actual_vs_predicted_aqi.png")