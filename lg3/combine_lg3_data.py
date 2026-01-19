import argparse
import os
import re
import shutil
from datetime import datetime

import pandas as pd


def parse_date_from_filename(path):
    m = re.search(r"_(\d{8})\.csv$", os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def copy_with_prefix(df, src_path, dst_dir, prefix):
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    dst_name = f"{prefix}_{name}{ext}"
    dst_path = os.path.join(dst_dir, dst_name)
    if not os.path.exists(dst_path):
        df.to_csv(dst_path, index=False)
        return dst_path
    idx = 1
    while True:
        dst_name = f"{prefix}_{name}_{idx}{ext}"
        dst_path = os.path.join(dst_dir, dst_name)
        if not os.path.exists(dst_path):
            df.to_csv(dst_path, index=False)
            return dst_path
        idx += 1


def should_skip(path, source, exclude_month):
    if source != "elec1_f2" or exclude_month is None:
        return False
    dt = parse_date_from_filename(path)
    if dt is None:
        return False
    return dt.month == exclude_month


def read_csv_remove_nulls(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if b"\x00" in raw:
            print(f"[WARN] Null bytes detected and removed in: {os.path.basename(path)}")
            raw = raw.replace(b"\x00", b"")
        return pd.read_csv(pd.io.common.BytesIO(raw))
    except Exception as exc:
        print(f"[WARN] Failed to read {os.path.basename(path)}: {exc}")
        return None


def drop_unnamed(df):
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]


def build_timestamp(df, src_path):
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
    if "Time" not in df.columns:
        return None
    dt = parse_date_from_filename(src_path)
    if dt is None:
        return None
    df["Timestamp"] = pd.to_datetime(
        dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str), errors="coerce"
    )
    return df


def filter_and_normalize(df, src_path):
    if df is None:
        return None
    df = drop_unnamed(df)
    df = build_timestamp(df, src_path)
    if df is None or "Timestamp" not in df.columns:
        print(f"[WARN] Skipping {os.path.basename(src_path)}: Missing timestamp/Time column.")
        return None
    df = df.dropna(subset=["Timestamp"])
    return df


def combine_sources(args):
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)

    out_ereport = os.path.join(args.output_dir, "EREPORT")
    out_smartcare = os.path.join(args.output_dir, "SMARTCARE")
    os.makedirs(out_ereport, exist_ok=True)
    os.makedirs(out_smartcare, exist_ok=True)

    totals = {"EREPORT": 0, "SMARTCARE": 0}
    skipped = {"EREPORT": 0, "SMARTCARE": 0}
    for source in args.sources:
        source_root = os.path.join(args.data_root, source)
        for subdir, out_dir in [("EREPORT", out_ereport), ("SMARTCARE", out_smartcare)]:
            src_dir = os.path.join(source_root, subdir)
            if not os.path.isdir(src_dir):
                print(f"[WARN] Missing {src_dir}, skipping.")
                continue
            for name in sorted(os.listdir(src_dir)):
                if not name.endswith(".csv"):
                    continue
                src_path = os.path.join(src_dir, name)
                if should_skip(src_path, source, args.exclude_elec1_month):
                    skipped[subdir] += 1
                    continue
                df = read_csv_remove_nulls(src_path)
                df = filter_and_normalize(df, src_path)
                if df is None or df.empty:
                    skipped[subdir] += 1
                    continue
                copy_with_prefix(df, src_path, out_dir, source)
                totals[subdir] += 1

    print(f"[DONE] EREPORT: copied {totals['EREPORT']}, skipped {skipped['EREPORT']}")
    print(f"[DONE] SMARTCARE: copied {totals['SMARTCARE']}, skipped {skipped['SMARTCARE']}")


def main():
    parser = argparse.ArgumentParser(description="Combine LG3 sources into a single EREPORT/SMARTCARE directory.")
    parser.add_argument(
        "--data_root", type=str, default="data", help="Root directory containing source folders."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="elec1_f2,ohsung_f2,snu",
        help="Comma-separated source folder names under data_root.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/combined", help="Where to write combined EREPORT/SMARTCARE."
    )
    parser.add_argument(
        "--exclude_elec1_month",
        type=int,
        default=11,
        help="Month (1-12) to exclude for elec1_f2 based on filename date.",
    )
    args = parser.parse_args()
    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    combine_sources(args)


if __name__ == "__main__":
    main()
