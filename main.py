import pandas as pd
import numpy as np
df=pd.read_csv("delhi_aqi.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(numeric_cols)

if df.isnull().sum().sum()==0:
    print("No missing value")
else:
    print("Missing value: ", df.isnull().sum().sum())

# checking duplication
date_dups = df.duplicated(subset="date").sum()
if df.duplicated().sum()==0 and date_dups==0:
    print("No duplicated row")
else:
    print(f"duplicated row: {df.duplicated().sum()}")
    print(f"duplicate timestamps: {date_dups}")

neg_found = False
for col in numeric_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        neg_found = True
        print(f"{col}:{neg} negative values")
        print(df[df[col] < 0][["date", col]].head(5).to_string())
if not neg_found:
    print("No negative values found.")
    
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    if zero_count == 0:
        print(f"  {col}: 0 zeros")
        continue
    zero_idx = df[df[col] == 0].index.to_numpy()
    runs = np.split(zero_idx, np.where(np.diff(zero_idx) != 1)[0] + 1)
    long_runs = [r for r in runs if len(r) >= 5]

    tag = " " if long_runs else ""
    print(f"\n{tag}{col}: {zero_count} zeros ({zero_pct:.1f}%), "
          f"{len(runs)} run(s), {len(long_runs)} run(s) ≥ {5} consecutive")

    if long_runs:
        print(f"  Longest consecutive zero runs (top 3):")
        sorted_runs = sorted(long_runs, key=len, reverse=True)[:3]
        for r in sorted_runs:
            start = df.loc[r[0], "date"]
            end = df.loc[r[-1], "date"]
            print(f"    {len(r)} hours: {start} → {end}")
            
time_diffs = df["date"].diff().dropna()
expected = pd.Timedelta("1h")
gap_mask = time_diffs > expected
gaps = time_diffs[gap_mask]

print(f"Total gaps (> 1h): {len(gaps)}")
if len(gaps) == 0:
    print("No time gaps found — data is continuous.")
else:
    print(f"\n{'Before gap':<30} {'After gap':<30} {'Duration'}")
    print("-" * 75)
    for idx, duration in gaps.sort_values(ascending=False).items():
        before = df.loc[idx - 1, "date"]
        after  = df.loc[idx, "date"]
        missing_hrs = int(duration.total_seconds() / 3600) - 1
        print(f"{str(before):<30} {str(after):<30} {str(duration)}  ({missing_hrs} missing hrs)")

print("\n" + "─" * 60)
print(f"7. SENSOR FREEZE  (≥ {5} consecutive identical values)")
print("─" * 60)

freeze_found = False
for col in numeric_cols:
    same_as_prev = df[col] == df[col].shift(1)
    groups = same_as_prev.groupby((same_as_prev != same_as_prev.shift()).cumsum())
    runs = groups.sum()
    max_run = int(runs.max())

    if max_run >= 5:
        freeze_found = True
        # Find where the longest run occurs
        long_groups = runs[runs >= 5]
        group_ids = long_groups.index.tolist()
        cumsum_series = (same_as_prev != same_as_prev.shift()).cumsum()
        print(f"\n  ⚠️  {col}: max consecutive identical = {max_run}")
        count = 0
        for gid in group_ids[:3]:
            block = df[cumsum_series == gid]
            if len(block) >= 5:
                val = block[col].iloc[0]
                start = block["date"].iloc[0]
                end   = block["date"].iloc[-1]
                print(f"Value {val:.2f} frozen for {len(block)} hours: {start} → {end}")
                count += 1
    else:
        print(f"{col}: OK (max repeat = {max_run}) ")

if not freeze_found:
    print("No sensor freeze detected.")
# print(df.head())