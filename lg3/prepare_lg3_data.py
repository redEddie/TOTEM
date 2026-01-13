import argparse
import csv
import glob
import os
import re
import pdb
import json

import numpy as np
import pandas as pd


EREPORT_DEFAULT_COLS = [
    "Capa_Cooling",
    "MFR_068",
    "Rop",
    "Comp1 Hz_1",
    "Comp1 Hz_0",
    "VAP_Entha",
    "LIQ_Entha",
    "cycle",
    "HighP",
    "LowP",
    "Tcond",
    "SCEEV_M",
]

SMARTCARE_DEFAULT_COLS = [
    "Tset",
    "Tid",
    "Hid",
    "Low P",
    "High P",
    "Power",
    "Tpip_in",
    "Frun",
]


def parse_date_from_filename(path):
    m = re.search(r"_(\d{8})\.csv$", os.path.basename(path))
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def drop_unnamed(df):
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]


def load_emeter(emeter_dir, cols):
    files = sorted(glob.glob(os.path.join(emeter_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No EMETER CSV files in {emeter_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df = drop_unnamed(df)
        date = parse_date_from_filename(path)
        if date is None or "Time" not in df.columns:
            raise ValueError(f"Missing date or Time column in {path}")
        df["Timestamp"] = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
            errors="coerce",
        )
        keep_cols = ["Timestamp"] + [c for c in cols if c in df.columns]
        df = df[keep_cols]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return out


def load_ereport(ereport_dir, cols):
    files = sorted(glob.glob(os.path.join(ereport_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No EREPORT CSV files in {ereport_dir}")
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            line_no, expected, got = find_bad_csv_line(path)
            message = (
                f"EREPORT parse error in {path} at line {line_no}: "
                f"expected {expected} fields, got {got}.\n"
                "Enter pdb to inspect. Continue to re-read and skip bad lines."
            )
            print(message)
            skip_bad_lines = True
            pdb.set_trace()
            if not skip_bad_lines:
                raise ValueError("Aborted due to EREPORT parse error.")
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        df = drop_unnamed(df)
        date = parse_date_from_filename(path)
        if date is None or "Time" not in df.columns:
            raise ValueError(f"Missing date or Time column in {path}")
        df["Timestamp"] = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
            errors="coerce",
        )
        keep_cols = ["Timestamp"] + [c for c in cols if c in df.columns]
        df = df[keep_cols]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return out


def load_smartcare(smartcare_dir, cols):
    files = sorted(glob.glob(os.path.join(smartcare_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No SMARTCARE CSV files in {smartcare_dir}")
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            line_no, expected, got = find_bad_csv_line(path)
            message = (
                f"SMARTCARE parse error in {path} at line {line_no}: "
                f"expected {expected} fields, got {got}.\n"
                "Enter pdb to inspect. Continue to re-read and skip bad lines."
            )
            print(message)
            skip_bad_lines = True
            pdb.set_trace()
            if not skip_bad_lines:
                raise ValueError("Aborted due to SMARTCARE parse error.")
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        df = drop_unnamed(df)
        date = parse_date_from_filename(path)
        if date is None or "Time" not in df.columns:
            raise ValueError(f"Missing date or Time column in {path}")
        df["Timestamp"] = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
            errors="coerce",
        )
        if "Auto Id" not in df.columns:
            raise ValueError(f"Missing Auto Id column in {path}")
        keep_cols = ["Timestamp", "Auto Id"] + [c for c in cols if c in df.columns]
        df = df[keep_cols]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return out


def find_bad_csv_line(path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        expected = len(header)
        for idx, row in enumerate(reader, start=2):
            if len(row) != expected:
                return idx, expected, len(row)
    return None, expected, None


def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resample_emeter(df, freq, cols):
    df = to_numeric(df, cols)
    df = df.set_index("Timestamp")
    df = df[cols].resample(freq).mean()
    return df


def resample_ereport(df, freq, cols):
    df = to_numeric(df, cols)
    df = df.set_index("Timestamp")
    df = df[cols].resample(freq).mean()
    return df


def resample_smartcare(df, freq, cols, smooth_window, wide=True):
    df = to_numeric(df, cols)
    out_frames = []
    for unit_id, g in df.groupby("Auto Id"):
        g = g.set_index("Timestamp")[cols].sort_index()
        g = g.resample(freq).mean()
        if smooth_window > 1:
            g = g.rolling(window=smooth_window, min_periods=1).mean()
        if wide:
            g = g.add_prefix(f"sc_{int(unit_id)}_")
        else:
            g["unit_id"] = unit_id
        out_frames.append(g)
    if wide:
        return pd.concat(out_frames, axis=1)
    return pd.concat(out_frames, axis=0).reset_index()


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

    np.save(os.path.join(out_dir, f"{prefix}_train.npy"), train.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_val.npy"), val.to_numpy(dtype=np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_test.npy"), test.to_numpy(dtype=np.float32))


def parse_cols(arg, default_cols):
    if not arg:
        return default_cols
    return [c.strip() for c in arg.split(",") if c.strip()]


def filter_by_months(df, months):
    if not months:
        return df
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df[~df["Timestamp"].dt.month.isin(months)]
    return df


def filter_month_weekends(df, month):
    if month is None:
        return df
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    is_month = df["Timestamp"].dt.month == month
    is_weekend = df["Timestamp"].dt.weekday >= 5
    return df[~(is_month & is_weekend)]


def filter_by_json(df, json_path):
    if not json_path:
        return df
    with open(json_path, "r") as f:
        payload = json.load(f)
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Exclude specific dates (YYYY-MM-DD)
    dates = payload.get("exclude_dates", [])
    if dates:
        dates = pd.to_datetime(dates, errors="coerce").date
        df = df[~df["Timestamp"].dt.date.isin(dates)]

    # Exclude ranges [{"start": "...", "end": "..."}]
    for rng in payload.get("exclude_ranges", []):
        start = pd.to_datetime(rng.get("start"), errors="coerce")
        end = pd.to_datetime(rng.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        df = df[~((df["Timestamp"] >= start) & (df["Timestamp"] <= end))]

    return df


def filter_by_time_range(df, start_time, end_time):
    if not start_time or not end_time:
        return df
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    start = pd.to_datetime(start_time, format="%H:%M").time()
    end = pd.to_datetime(end_time, format="%H:%M").time()
    times = df["Timestamp"].dt.time
    if start < end:
        mask = (times >= start) & (times < end)
    else:
        mask = (times >= start) | (times < end)
    return df[~mask]


def main():
    parser = argparse.ArgumentParser(description="Prepare LG3 EREPORT+SMARTCARE data")
    parser.add_argument("--ereport_dir", type=str, default="lg3/data/EREPORT")
    parser.add_argument("--smartcare_dir", type=str, default="lg3/data/SMARTCARE")
    parser.add_argument("--output_dir", type=str, default="lg3/data/processed")
    parser.add_argument("--freq", type=str, default="1min")
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--ereport_cols", type=str, default="")
    parser.add_argument("--smartcare_cols", type=str, default="")
    parser.add_argument(
        "--exclude_months",
        type=str,
        default="",
        help="comma-separated months to exclude, e.g. '8,9'",
    )
    parser.add_argument(
        "--exclude_weekend_month",
        type=int,
        default=8,
        help="exclude weekends for a specific month (1-12)",
    )
    parser.add_argument(
        "--exclude_dates_json",
        type=str,
        default="",
        help="path to json with exclude_dates/exclude_ranges",
    )
    parser.add_argument(
        "--exclude_time_range",
        type=str,
        default="",
        help="exclude time range in HH:MM-HH:MM, e.g. 00:00-06:00",
    )
    parser.add_argument(
        "--smartcare_mode",
        type=str,
        default="per_unit",
        choices=["per_unit", "wide", "none"],
        help="per_unit creates per-unit datasets without Auto Id column",
    )
    args = parser.parse_args()

    er_cols = parse_cols(args.ereport_cols, EREPORT_DEFAULT_COLS)
    sc_cols = parse_cols(args.smartcare_cols, SMARTCARE_DEFAULT_COLS)
    exclude_months = [int(m) for m in args.exclude_months.split(",") if m.strip()]
    exclude_weekend_month = args.exclude_weekend_month
    exclude_dates_json = args.exclude_dates_json
    exclude_time_range = args.exclude_time_range

    er_raw = load_ereport(args.ereport_dir, er_cols)
    sc_raw = load_smartcare(args.smartcare_dir, sc_cols)

    er_raw = filter_by_months(er_raw, exclude_months)
    sc_raw = filter_by_months(sc_raw, exclude_months)
    er_raw = filter_month_weekends(er_raw, exclude_weekend_month)
    sc_raw = filter_month_weekends(sc_raw, exclude_weekend_month)
    er_raw = filter_by_json(er_raw, exclude_dates_json)
    sc_raw = filter_by_json(sc_raw, exclude_dates_json)
    if exclude_time_range:
        start, end = [s.strip() for s in exclude_time_range.split("-", 1)]
        er_raw = filter_by_time_range(er_raw, start, end)
        sc_raw = filter_by_time_range(sc_raw, start, end)

    base = resample_ereport(er_raw, args.freq, er_cols).dropna(how="any")

    if args.smartcare_mode == "none":
        train, val, test = time_split(base, args.train_ratio, args.val_ratio)
        save_splits(args.output_dir, "lg3", train, val, test)
        return

    if args.smartcare_mode == "wide":
        sc = resample_smartcare(sc_raw, args.freq, sc_cols, args.smooth_window, wide=True)
        merged = base.join(sc, how="inner").dropna(how="any")
        train, val, test = time_split(merged, args.train_ratio, args.val_ratio)
        save_splits(args.output_dir, "lg3", train, val, test)
        return

    # per-unit mode: create per-unit datasets without Auto Id column
    unit_dir = os.path.join(args.output_dir, "smartcare_units")
    os.makedirs(unit_dir, exist_ok=True)
    for unit_id, g in sc_raw.groupby("Auto Id"):
        g = g.copy()
        g = to_numeric(g, sc_cols)
        g = g.set_index("Timestamp")[sc_cols].resample(args.freq).mean()
        if args.smooth_window > 1:
            g = g.rolling(window=args.smooth_window, min_periods=1).mean()
        merged = base.join(g, how="inner").dropna(how="any")
        train, val, test = time_split(merged, args.train_ratio, args.val_ratio)
        unit_out = os.path.join(unit_dir, f"unit_{int(unit_id)}")
        save_splits(unit_out, "lg3", train, val, test)


if __name__ == "__main__":
    main()
