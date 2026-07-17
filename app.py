import sys
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)


MIN_FREEZE_RUN = 5
IQR_MULTIPLIER = 2.5
NIGHT_O3_THRESHOLD = 40
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6


def fix_sensor_freeze(df, col, min_run=5):
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


def fig_to_base64(fig):
    """Converts a matplotlib figure directly into a base64 string for HTML rendering."""
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def run_pipeline():

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:

        df = pd.read_csv("delhi_aqi.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        print("Shape before anomaly fixes:", df.shape)

        if df.isnull().sum().sum() == 0:
            print("\nFix-1\n\tNo missing value")
        else:
            print("Missing value: ", df.isnull().sum().sum())

        date_dups = df.duplicated(subset="date").sum()
        if df.duplicated().sum() == 0 and date_dups == 0:
            print("\nFix-2\n\tNo duplicated row")
        else:
            print(f"duplicated row: {df.duplicated().sum()} | duplicate timestamps: {date_dups}")

        neg_found = False
        for col in numeric_cols:
            neg = (df[col] < 0).sum()
            if neg > 0:
                neg_found = True
                print(f"{col}:{neg} negative values")
        if not neg_found:
            print("\nFix-3\n\tNo negative values found.")


        full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="1h")
        df = df.set_index("date").reindex(full_range)
        df.index.name = "date"
        df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        df = df.reset_index()


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


        print("\nFix 5 — Sensor freeze (pass 1):")
        for col in ["no", "o3"]:
            df = fix_sensor_freeze(df, col, MIN_FREEZE_RUN)

        caps_25 = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            caps_25[col] = Q3 + IQR_MULTIPLIER * (Q3 - Q1)

        print("\nFix 6 — Nighttime O3 correction:")
        df = df.set_index("date")
        df["hour"] = df.index.hour
        night_mask = ((df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)) & (
                    df["o3"] > NIGHT_O3_THRESHOLD)
        df.loc[night_mask, "o3"] = np.nan
        df["o3"] = df["o3"].interpolate(method="time")
        night_hours = (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
        df.loc[night_hours & (df["o3"] > NIGHT_O3_THRESHOLD), "o3"] = NIGHT_O3_THRESHOLD
        df = df.drop(columns="hour").reset_index()

        print("\nFix 7 — Smooth cap plateaus + apply 2.5x ceiling:")
        df = df.set_index("date")
        for col in numeric_cols:
            df[col] = df[col].clip(upper=caps_25[col])
        df = df.reset_index()

        print(f"\npm2_5 ~ pm10 corr     : {df['pm2_5'].corr(df['pm10']):.3f}")
        print(f"Final data shape       : {df.shape}")


        df["hour"] = df["date"].dt.hour
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["season"] = df["month"].apply(get_season)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        pollutant_columns = ["pm2_5", "pm10", "no", "no2", "nh3", "co", "so2", "o3"]
        for col in pollutant_columns:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1)
                df[f"{col}_lag24"] = df[col].shift(24)

        df = df.iloc[24:].reset_index(drop=True)
        df["pm2_5_raw"] = df["pm2_5"]
        df["pm10_raw"] = df["pm10"]

        print("\nFeature Engineering Completed Successfully")


        if "date" in df.columns:
            df = df.drop("date", axis=1)

        df["pm25_aqi"] = df["pm2_5_raw"].apply(get_pm25_subindex)
        df["pm10_aqi"] = df["pm10_raw"].apply(get_pm10_subindex)
        df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1).round(0)
        df = df.drop(columns=["pm25_aqi", "pm10_aqi", "pm2_5_raw", "pm10_raw"])

        X = df.drop(columns=["aqi", "pm2_5", "pm10"], errors="ignore")
        y = df["aqi"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestRegressor(n_estimators=50, random_state=42,
                                      n_jobs=-1)  # slightly lowered trees for fast web render
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\nModel Evaluation Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  R-squared (R2) Score: {r2:.2f}")


        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=True).tail(15)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
        ax1.set_xlabel("Importance Score")
        ax1.set_title("Top 15 Random Forest Feature Importances")
        plot1_b64 = fig_to_base64(fig1)


        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_test, predictions, color="steelblue", alpha=0.6, edgecolors="black", s=30, label="Predictions")
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2,
                 label="Perfect Prediction")
        ax2.set_xlabel("Actual AQI")
        ax2.set_ylabel("Predicted AQI")
        ax2.set_title("Actual vs Predicted AQI")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plot2_b64 = fig_to_base64(fig2)

    finally:

        terminal_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    return terminal_output, plot1_b64, plot2_b64


@app.route("/")
def home():
    output_text, graph1, graph2 = run_pipeline()
    return render_template("index.html", output=output_text, chart1=graph1, chart2=graph2)


if __name__ == "__main__":
    import os


    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    app.template_folder = template_dir
    app.run(debug=True, port=5000)