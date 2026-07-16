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
from sklearn.preprocessing import StandardScaler

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
    then interpolate over time."""
    same_as_prev = df[col] == df[col].shift(1)
    cumsum_series = (same_as_prev != same_as_prev.shift()).cumsum()
    groups = same_as_prev.groupby(cumsum_series)
    run_lengths = groups.sum()  # length of each run
    frozen_group_ids = run_lengths[run_lengths >= min_run].index

    n_replaced = 0
    for gid in frozen_group_ids:
        mask = cumsum_series == gid
        df.loc[mask, col] = np.nan
        n_replaced += mask.sum()

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
print("\n  Post-smooth freeze cleanup:")
for col in numeric_cols:
    s = df[col] == df[col].shift(1)
    cumsum_series = (s != s.shift()).cumsum()
    runs = s.groupby(cumsum_series).sum()
    long_groups = runs[runs >= 5].index.tolist()

    if not long_groups:
        print(f"  {col}: OK")
        continue

    n_replaced = 0
    for gid in long_groups:
        mask = cumsum_series == gid
        df.loc[mask, col] = np.nan
        n_replaced += mask.sum()

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

df_tmp = df.set_index("date")
max_freezes = {}
for col in numeric_cols:
    series = df_tmp[col].copy()
    if col == "o3":
        night_hrs = (df_tmp.index.hour >= NIGHT_START_HOUR) | (df_tmp.index.hour < NIGHT_END_HOUR)
        series[night_hrs & (series == NIGHT_O3_THRESHOLD)] = np.nan
    s = series == series.shift(1)
    runs = s.groupby((s != s.shift()).cumsum()).sum()
    max_freezes[col] = int(runs.max())

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
    big_drop = (df[col] - next_val) / df[col] > 0.40
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

for col in ["pm2_5", "pm10", "co"]:
    if col in df.columns:
        df[f"{col}_rolling3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_rolling24"] = df[col].rolling(window=24, min_periods=1).mean()

df = df.bfill()

# --- Scaling ---
scale_columns = pollutant_columns + [
    col for col in df.columns if "_lag" in col or "_rolling" in col
]

scaler = StandardScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])

print("\nFeature Engineering Completed Successfully")
print("Shape after Feature Engineering:", df.shape)

print("\nNew Features Added:")
print([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season_sin", "season_cos",
    "weekday_sin", "weekday_cos",
])

print("\nScaled Columns:")
print(scale_columns)

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

df["pm25_aqi"] = df["pm2_5"].apply(get_pm25_subindex)
df["pm10_aqi"] = df["pm10"].apply(get_pm10_subindex)
df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1).round(0)
df = df.drop(columns=["pm25_aqi", "pm10_aqi"])

target_col = "aqi" if "aqi" in df.columns else "pm2_5"
X = df.drop(columns=[target_col, "pm2_5", "pm10"], errors="ignore")
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# ============================================================
# Actual vs Predicted AQI
# ============================================================
plt.figure(figsize=(8, 8))

# Scatter plot
plt.scatter(
    y_test,
    predictions,
    color="steelblue",
    alpha=0.7,
    edgecolors="black",
    s=45,
    label="Predictions"
)

# Perfect prediction line (y = x)
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

# Same scale on both axes
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Labels and title
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Random Forest)")

# Display evaluation metrics
plt.text(
    0.05,
    0.95,
    f"MAE = {mae:.2f}\nR² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(facecolor="white", edgecolor="gray")
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

print("Chart successfully saved as actual_vs_predicted.png!")