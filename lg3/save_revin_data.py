import argparse
import json
import glob
import os

import numpy as np
import pandas as pd
import torch

from lg3.lib.models.revin import RevIN


def load_split(path, cols=None):
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if cols:
        keep = [c for c in cols if c in df.columns]
        df = df[keep]
    else:
        df = df.select_dtypes(include=[np.number])
    df = df.dropna(how="any")
    return df


def build_sequences(values, seq_len, pred_len):
    total = len(values)
    max_start = total - (seq_len + pred_len) + 1
    if max_start <= 0:
        raise ValueError("Not enough rows to build sequences.")
    x = np.empty((max_start, seq_len, values.shape[1]), dtype=np.float32)
    y = np.empty((max_start, pred_len, values.shape[1]), dtype=np.float32)
    for i in range(max_start):
        x[i] = values[i : i + seq_len]
        y[i] = values[i + seq_len : i + seq_len + pred_len]
    return x, y


def revin_normalize(arr, revin_layer, batch_size, device):
    outputs = []
    for i in range(0, arr.shape[0], batch_size):
        batch = torch.tensor(arr[i : i + batch_size], dtype=torch.float32, device=device)
        out = revin_layer(batch, "norm").detach().cpu().numpy()
        outputs.append(out)
    return np.concatenate(outputs, axis=0)


def flatten_sensors(arr):
    return np.swapaxes(arr, 1, 2).reshape((-1, arr.shape[1]))


def process_split(df, seq_len, pred_len, revin_x, revin_y, batch_size, device):
    values = df.to_numpy(dtype=np.float32)
    x, y = build_sequences(values, seq_len, pred_len)
    x_norm = revin_normalize(x, revin_x, batch_size, device)
    y_norm = revin_normalize(y, revin_y, batch_size, device)
    if seq_len != pred_len:
        raise ValueError("seq_len must equal pred_len to flatten sensors.")
    x_flat = flatten_sensors(x_norm)
    y_flat = flatten_sensors(y_norm)
    return x_flat, y_flat


def load_unit_splits(input_dir, split):
    unit_root = os.path.join(input_dir, "smartcare_units")
    if not os.path.isdir(unit_root):
        return []
    unit_dirs = sorted(
        d for d in glob.glob(os.path.join(unit_root, "unit_*")) if os.path.isdir(d)
    )
    return [os.path.join(d, f"lg3_{split}.csv") for d in unit_dirs]


def main():
    parser = argparse.ArgumentParser(description="LG3 save_revin_data")
    parser.add_argument("--input_dir", type=str, default="lg3/data/processed")
    parser.add_argument("--output_dir", type=str, default="lg3/data/revin")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cols", type=str, default="")
    args = parser.parse_args()

    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    cols = cols if cols else None

    train_paths = load_unit_splits(args.input_dir, "train")
    val_paths = load_unit_splits(args.input_dir, "val")
    test_paths = load_unit_splits(args.input_dir, "test")

    if not train_paths:
        train_paths = [os.path.join(args.input_dir, "lg3_train.csv")]
        val_paths = [os.path.join(args.input_dir, "lg3_val.csv")]
        test_paths = [os.path.join(args.input_dir, "lg3_test.csv")]
        missing = [p for p in train_paths + val_paths + test_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "No prepared splits found. Expected lg3_train.csv/lg3_val.csv/lg3_test.csv in input_dir."
            )

    sample_df = load_split(train_paths[0], cols=cols)
    num_features = sample_df.shape[1]
    feature_names = list(sample_df.columns)
    revin_x = RevIN(num_features=num_features, affine=False, subtract_last=False).to(device)
    revin_y = RevIN(num_features=num_features, affine=False, subtract_last=False).to(device)

    def process_paths(paths):
        xs, ys = [], []
        for path in paths:
            df = load_split(path, cols=cols)
            x, y = process_split(df, args.seq_len, args.pred_len, revin_x, revin_y, args.batch_size, device)
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    x_train, y_train = process_paths(train_paths)
    x_val, y_val = process_paths(val_paths)
    x_test, y_test = process_paths(test_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    feature_map = {
        "features": feature_names,
        "index_by_feature": {name: idx for idx, name in enumerate(feature_names)},
    }
    with open(os.path.join(args.output_dir, "feature_map.json"), "w") as fh:
        json.dump(feature_map, fh, indent=2)
    np.save(os.path.join(args.output_dir, "train_data_x.npy"), x_train)
    np.save(os.path.join(args.output_dir, "val_data_x.npy"), x_val)
    np.save(os.path.join(args.output_dir, "test_data_x.npy"), x_test)
    np.save(os.path.join(args.output_dir, "train_data_y.npy"), y_train)
    np.save(os.path.join(args.output_dir, "val_data_y.npy"), y_val)
    np.save(os.path.join(args.output_dir, "test_data_y.npy"), y_test)


if __name__ == "__main__":
    main()
