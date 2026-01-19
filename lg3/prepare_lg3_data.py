import argparse
import csv
import glob
import os
import re
import pdb
import json
import io  # Added
import numpy as np
import pandas as pd

# --- Utility Functions (Kept and Added) ---

def read_csv_remove_nulls(path):
    """Reads a CSV file after removing any null bytes."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if b"\x00" in raw:
            print(f"[WARN] Null bytes detected and removed in: {os.path.basename(path)}")
            raw = raw.replace(b"\x00", b"")
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        try:
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            print(f"[DEBUG] Total lines: {len(lines)}")
            if lines:
                header = lines[0]
                header_fields = header.split(",")
                print(f"[DEBUG] Header fields: {len(header_fields)}")
            msg = str(e)
            m = re.search(r"line (\\d+)", msg)
            if m:
                line_no = int(m.group(1))
                if 1 <= line_no <= len(lines):
                    bad_line = lines[line_no - 1]
                    print(f"[DEBUG] Error line {line_no} preview: {bad_line[:200]}")
        except Exception as debug_e:
            print(f"[DEBUG] Failed to extract error context: {debug_e}")
        print(f"[SKIP] Skipping file due to read/parse failure: {os.path.basename(path)}")
        return None

def parse_date_from_filename(path):
    m = re.search(r"_(\d{8})\.csv$", os.path.basename(path))
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")

def drop_unnamed(df):
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]

def load_data(data_dir, required_cols_list, use_auto_id=False): # Changed arg name
    """Generic data loader for EREPORT and SMARTCARE."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in files:
        df = read_csv_remove_nulls(path)
        if df is None:
            print(f"[SKIP] Skipped unreadable file: {os.path.basename(path)}")
            continue

        df = drop_unnamed(df)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        else:
            date = parse_date_from_filename(path)
            if date is None or "Time" not in df.columns:
                print(f"[WARN] Skipping {os.path.basename(path)}: Missing date or 'Time' column.")
                continue
            df["Timestamp"] = pd.to_datetime(
                date.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
                errors="coerce",
            )
        
        cols_to_keep = ["Timestamp"]
        if use_auto_id:
            cols_to_keep.append("Auto Id")
        
        # Ensure 'Auto Id' is in df.columns if use_auto_id is True
        if use_auto_id and "Auto Id" not in df.columns:
            print(f"[WARN] Skipping {os.path.basename(path)}: 'Auto Id' column missing but required.")
            continue

        col_aliases = {
            "VAP_Entha": "VAP_Entha_R32",
            "LIQ_Entha": "LIQ_Entha_R32",
        }

        resolved_cols = []
        missing_cols = []
        used_aliases = {}
        for col in required_cols_list:
            if col in df.columns:
                resolved_cols.append(col)
            else:
                alias = col_aliases.get(col)
                if alias and alias in df.columns:
                    resolved_cols.append(col)
                    used_aliases[col] = alias
                else:
                    missing_cols.append(col)

        if used_aliases:
            df = df.rename(columns={v: k for k, v in used_aliases.items()})

        final_cols = cols_to_keep + resolved_cols

        if missing_cols:
            print(f"[WARN] Missing columns {missing_cols} in {os.path.basename(path)}. Some data might be incomplete.")
            print(f"[DEBUG] Available columns: {list(df.columns)}")
            if used_aliases:
                print(f"[DEBUG] Used aliases: {used_aliases}")
            if any(col in missing_cols for col in ["VAP_Entha", "LIQ_Entha"]):
                print(f"[WARN] Skipping {os.path.basename(path)} due to missing Entha columns.")
                continue

        df = df[final_cols]
        frames.append(df)

    if not frames:
        raise ValueError(f"No valid data could be loaded from {data_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return out

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def resample_ereport(df, freq, cols):
    df = to_numeric(df, cols)
    df = df.set_index("Timestamp")
    resampled = df[cols].resample(freq, label="left", closed="left").mean()
    return resampled

def create_time_features(dt_index):
    """Creates time-based features from a DatetimeIndex."""
    df_time = pd.DataFrame(index=dt_index)
    hour_of_day = df_time.index.hour.astype(int)
    df_time["day_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df_time["day_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    try:
        freq_timedelta = pd.to_timedelta(dt_index.freqstr)
        minutes_per_unit = freq_timedelta.total_seconds() / 60.0
        if minutes_per_unit == 0:
            minutes_per_unit = 1
        steps_per_week = (7 * 24 * 60) / minutes_per_unit
    except Exception:
        steps_per_week = 7 * 24
        print(f"[WARN] Could not accurately determine steps per week from frequency '{dt_index.freqstr}'. Assuming hourly.")

    k = np.arange(len(df_time), dtype=float)
    df_time["week_sin"] = np.sin(2 * np.pi * k / steps_per_week)
    df_time["week_cos"] = np.cos(2 * np.pi * k / steps_per_week)
    return df_time


def load_holidays(path):
    if not path:
        return set()
    with open(path, "r") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        dates = []
        for key in ("weekends", "public_holidays"):
            dates.extend(data.get(key, []))
        return set(dates)
    return set(data)


def fft_reconstruct(y, k):
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    fft = np.fft.rfft(y)
    keep = np.zeros_like(fft)
    keep[0] = fft[0]
    if k > 0 and len(fft) > 1:
        mags = np.abs(fft)
        idx = np.argsort(mags[1:])[-k:] + 1
        keep[idx] = fft[idx]
    return np.fft.irfft(keep, n=n)


def compute_fourier_feature(series, freq, k, lookback_weeks=(1, 2, 3)):
    series = series.sort_index()
    day_steps = int((24 * 60) / pd.to_timedelta(freq).total_seconds() * 60)
    unique_dates = pd.to_datetime(series.index.date).unique()
    values_by_date = {}
    for d in unique_dates:
        day_start = pd.to_datetime(d)
        day_index = pd.date_range(day_start, day_start + pd.Timedelta(days=1) - pd.to_timedelta(freq), freq=freq)
        day_series = series.loc[day_start:day_start + pd.Timedelta(days=1) - pd.to_timedelta(freq)].reindex(day_index)
        if day_series.isna().any():
            continue
        if len(day_series) == day_steps:
            values_by_date[day_start.date()] = day_series.values

    fourier_series = pd.Series(index=series.index, dtype=float)
    for d in pd.to_datetime(series.index.date).unique():
        d_date = pd.to_datetime(d).date()
        prior_values = []
        for w in lookback_weeks:
            prior_date = d_date - pd.Timedelta(days=7 * w)
            vals = values_by_date.get(prior_date)
            if vals is None:
                prior_values = []
                break
            prior_values.append(vals)
        if not prior_values:
            continue
        avg_prior = np.mean(np.vstack(prior_values), axis=0)
        recon = fft_reconstruct(avg_prior, k)
        day_start = pd.to_datetime(d_date)
        day_index = pd.date_range(day_start, day_start + pd.Timedelta(days=1) - pd.to_timedelta(freq), freq=freq)
        fourier_series.loc[day_index] = recon
    return fourier_series

def preprocess_smartcare_cols(df_smart, freq, process_cols_list):
    """Aggregates specified SMARTCARE columns across all units, ignoring Auto Id."""
    
    # Only process columns specified in process_cols_list
    df_to_process = df_smart[["Timestamp"] + process_cols_list].copy().set_index("Timestamp")
    
    df_to_process["slot"] = df_to_process.index.floor(freq)
    
    # Process each column in process_cols_list
    aggregated_results = {}
    for col in process_cols_list:
        def mode_or_nan(series):
            mode_vals = series.dropna().mode()
            if mode_vals.empty:
                return np.nan
            return float(mode_vals.iloc[0])

        col_mode = df_to_process.groupby("slot")[col].apply(mode_or_nan)
        aggregated_results[f"{col}"] = col_mode

    return pd.DataFrame(aggregated_results)


def time_split(df, train_ratio, val_ratio):
    df = df.sort_index()
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def save_splits(out_dir, prefix, train, val, test):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, f"{prefix}_train.csv"))
    val.to_csv(os.path.join(out_dir, f"{prefix}_val.csv"))
    test.to_csv(os.path.join(out_dir, f"{prefix}_test.csv"))
    print(f"Saved train/val/test CSVs to {out_dir}")

    np.save(os.path.join(out_dir, f"{prefix}_train.npy"), train.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_val.npy"), val.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_test.npy"), test.to_numpy(dtype=np.float32))
    print(f"Saved train/val/test .npy files to {out_dir}")


def save_exog_splits(out_dir, prefix, train, val, test, feature_names):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_train_exog.npy"), train.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_val_exog.npy"), val.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_test_exog.npy"), test.to_numpy(dtype=np.float32))
    feature_map = {
        "features": feature_names,
        "index_by_feature": {name: idx for idx, name in enumerate(feature_names)},
    }
    with open(os.path.join(out_dir, f"{prefix}_exog_map.json"), "w") as fh:
        json.dump(feature_map, fh, indent=2)

def parse_cols(arg):
    if not arg:
        raise ValueError("Column list is required. Provide --ereport_cols and --smartcare_process_cols.")
    return [c.strip() for c in arg.split(",") if c.strip()]


def main():
    parser = argparse.ArgumentParser(description="Prepare LG3 aggregated data (no Auto Id)")
    parser.add_argument("--ereport_dir", type=str, default="data/elec1_f2/EREPORT")
    parser.add_argument("--smartcare_dir", type=str, default="data/elec1_f2/SMARTCARE")
    parser.add_argument("--output_dir", type=str, default="lg3/data/processed")
    parser.add_argument("--freq", type=str, default="15min")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--ereport_cols", 
        type=str, 
        default="",
        help="Comma-separated list of EREPORT columns to process."
    )
    # New argument for SMARTCARE columns to process
    parser.add_argument(
        "--smartcare_process_cols",
        type=str,
        default="Tod",
        help="Comma-separated list of SMARTCARE columns to process and aggregate."
    )
    parser.add_argument(
        "--exog_output_dir",
        type=str,
        default="lg3/data/exog",
        help="Where to save exogenous features as separate npy files.",
    )
    parser.add_argument(
        "--holiday_path",
        type=str,
        default="",
        help="Path to JSON list of YYYY-MM-DD holiday dates.",
    )
    parser.add_argument(
        "--fourier_col",
        type=str,
        default="Power",
        help="Column used to compute Fourier exogenous feature.",
    )
    parser.add_argument(
        "--fourier_k",
        type=int,
        default=15,
        help="Number of FFT frequencies to keep for Fourier feature.",
    )
    parser.add_argument(
        "--exclude_from_month",
        type=int,
        default=0,
        help="Exclude data with month >= this value (e.g., 10 removes Oct/Nov/Dec).",
    )
    # Args for Tod aggregation
    args = parser.parse_args()

    # --- 0. Path checks ---
    for label, path in [("EREPORT", args.ereport_dir), ("SMARTCARE", args.smartcare_dir)]:
        if not os.path.isdir(path):
            print(f"[ERROR] {label} directory not found: {path}")
        elif not any(name.endswith(".csv") for name in os.listdir(path)):
            print(f"[ERROR] {label} directory has no CSV files: {path}")

    existing_outputs = [
        os.path.join(args.output_dir, "lg3_train.csv"),
        os.path.join(args.output_dir, "lg3_val.csv"),
        os.path.join(args.output_dir, "lg3_test.csv"),
    ]
    if any(os.path.exists(p) for p in existing_outputs):
        print(f"[WARN] Existing prepared outputs found in: {args.output_dir}")
        pdb.set_trace()

    # --- 1. Load Data ---
    er_cols = parse_cols(args.ereport_cols)
    smartcare_process_cols = parse_cols(args.smartcare_process_cols)

    print("Loading EREPORT data...")
    er_raw = load_data(args.ereport_dir, er_cols)
    
    print(f"Loading SMARTCARE data (columns: {smartcare_process_cols})...")
    sc_raw = load_data(args.smartcare_dir, smartcare_process_cols, use_auto_id=False)
    
    # --- 2. Process and Resample Data ---
    print(f"Resampling EREPORT data to {args.freq}...")
    base_df = resample_ereport(er_raw, args.freq, er_cols)
    
    print(f"Aggregating SMARTCARE columns: {smartcare_process_cols}...")
    smartcare_agg_df = preprocess_smartcare_cols(
        sc_raw,
        args.freq,
        smartcare_process_cols
    )
    
    # --- 3. Create Exogenous Features ---
    print("Creating time features...")
    time_df = create_time_features(base_df.index)
    holidays = load_holidays(args.holiday_path)
    is_holiday = base_df.index.normalize().strftime("%Y-%m-%d").isin(holidays)
    fourier_series = compute_fourier_feature(
        base_df[args.fourier_col],
        args.freq,
        args.fourier_k,
    )
    exog_df = time_df.copy()
    exog_df["is_holiday"] = is_holiday.astype(int)
    exog_df["fourier"] = fourier_series

    # --- 4. Merge all dataframes ---
    print("Merging dataframes...")
    merged_df = base_df.join(smartcare_agg_df, how="left")
    
    # Drop rows where joins might have failed (especially for Tod)
    n_before = len(merged_df)
    merged_df = merged_df.dropna()
    exog_df = exog_df.reindex(merged_df.index)
    if args.exclude_from_month:
        merged_df = merged_df[merged_df.index.month < args.exclude_from_month]
        exog_df = exog_df[exog_df.index.month < args.exclude_from_month]
    n_after = len(merged_df)
    print(f"Dropped {n_before - n_after} rows with NaN values after merging.")
    
    print("\nFinal dataset columns:")
    print(merged_df.columns.tolist())
    print(f"Final dataset shape: {merged_df.shape}")

    # --- 5. Split and Save ---
    train, val, test = time_split(merged_df, args.train_ratio, args.val_ratio)
    print(f"Train samples: {len(train)}, Validation samples: {len(val)}, Test samples: {len(test)}")
    if not train.empty:
        print(f"Train range: {train.index.min()} -> {train.index.max()}")
    if not val.empty:
        print(f"Val range: {val.index.min()} -> {val.index.max()}")
    if not test.empty:
        print(f"Test range: {test.index.min()} -> {test.index.max()}")
    
    save_splits(args.output_dir, "lg3", train, val, test)
    exog_train, exog_val, exog_test = time_split(exog_df, args.train_ratio, args.val_ratio)
    save_exog_splits(args.exog_output_dir, "lg3", exog_train, exog_val, exog_test, list(exog_df.columns))

if __name__ == "__main__":
    main()
