import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_split_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna(how="any")
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


def build_lagged_sequences(
    values,
    pred_len,
    lag_days,
    lag_window_steps,
    steps_per_day,
    sep_steps,
    valid_start_idx=None,
    valid_end_idx=None,
):
    lag_steps = [d * steps_per_day for d in lag_days]
    max_required = max(
        lag_steps[i] + lag_window_steps[i] for i in range(len(lag_steps))
    )
    n = len(values)

    x_list = []
    y_list = []
    for t in range(max_required, n - pred_len + 1):
        if valid_start_idx is not None and t < valid_start_idx:
            continue
        if valid_end_idx is not None and (t + pred_len) > valid_end_idx:
            continue
        segments = []
        for idx, lag in enumerate(lag_steps):
            window_len = lag_window_steps[idx]
            start = t - lag - window_len
            end = t - lag
            segments.append(values[start:end])

        if sep_steps > 0:
            sep_pad = []
            for i, seg in enumerate(segments):
                sep_pad.append(seg)
                if i < len(segments) - 1:
                    last_val = seg[-1:]
                    sep_pad.append(np.repeat(last_val, sep_steps, axis=0))
            x_seq = np.concatenate(sep_pad, axis=0)
        else:
            x_seq = np.concatenate(segments, axis=0)

        y_seq = values[t : t + pred_len]
        x_list.append(x_seq)
        y_list.append(y_seq)

    if not x_list:
        raise ValueError("Not enough rows to build lagged sequences.")

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return x, y


class ExtractRawData:
    def __init__(self, args):
        self.args = args
        self.use_lagged = args.use_lagged
        self.lag_days = [int(x) for x in args.lag_days.split(",") if x.strip()]
        if not self.lag_days:
            self.lag_days = [0]
        self.lag_window_days = [int(x) for x in args.lag_window_days.split(",") if x.strip()]
        if not self.lag_window_days:
            self.lag_window_days = [1]
        if len(self.lag_window_days) == 1 and len(self.lag_days) > 1:
            self.lag_window_days = self.lag_window_days * len(self.lag_days)
        if len(self.lag_window_days) != len(self.lag_days):
            raise ValueError("lag_window_days must match lag_days length.")
        self.steps_per_day = args.steps_per_day
        self.sep_steps = args.sep_steps
        self.lag_window_steps = [d * self.steps_per_day for d in self.lag_window_days]

        sample_df = self._load_split_df("train")
        self.feature_names = sample_df.columns.tolist()

    def _load_split_df(self, split):
        unit_data_path = os.path.join(self.args.input_dir, "smartcare_units")
        all_files = glob.glob(os.path.join(unit_data_path, f"unit_*/lg3_{split}.csv"))
        if not all_files:
            split_path = os.path.join(self.args.input_dir, f"lg3_{split}.csv")
            if not os.path.exists(split_path):
                raise FileNotFoundError(
                    f"No data files found for split '{split}' in {unit_data_path} or {self.args.input_dir}"
                )
            all_files = [split_path]
        df_list = [load_split_csv(f) for f in all_files]
        return pd.concat(df_list, ignore_index=True)

    def _get_split(self, split):
        combined_df = self._load_split_df(split)
        values = combined_df.to_numpy(dtype=np.float32)
        if self.use_lagged:
            max_required = max(
                (self.lag_days[i] * self.steps_per_day) + self.lag_window_steps[i]
                for i in range(len(self.lag_days))
            )
            history_len = max_required

            if split == "train":
                values_all = values
                valid_start = 0
            else:
                prev_splits = ["train"] if split == "val" else ["train", "val"]
                prev_df = pd.concat(
                    [self._load_split_df(s) for s in prev_splits], ignore_index=True
                )
                prev_vals = prev_df.to_numpy(dtype=np.float32)
                if len(prev_vals) > history_len:
                    prev_vals = prev_vals[-history_len:]
                values_all = np.concatenate([prev_vals, values], axis=0)
                valid_start = len(prev_vals)

            valid_end = valid_start + len(values)
            x, y = build_lagged_sequences(
                values_all,
                self.args.pred_len,
                self.lag_days,
                self.lag_window_steps,
                self.steps_per_day,
                self.sep_steps,
                valid_start_idx=valid_start,
                valid_end_idx=valid_end,
            )
        else:
            x, y = build_sequences(values, self.args.seq_len, self.args.pred_len)
        return x, y

    def extract_data(self):
        if self.use_lagged:
            total_len = sum(self.lag_window_steps) + self.sep_steps * (
                len(self.lag_days) - 1
            )
            if total_len % self.args.compression_factor != 0:
                raise ValueError(
                    "Lagged input length must be divisible by compression_factor."
                )
        else:
            if self.args.seq_len % self.args.compression_factor != 0:
                raise ValueError("seq_len must be divisible by compression_factor.")
        if self.args.pred_len % self.args.compression_factor != 0:
            raise ValueError("pred_len must be divisible by compression_factor.")

        os.makedirs(self.args.save_path, exist_ok=True)
        pd.Series(self.feature_names).to_json(
            os.path.join(self.args.save_path, "feature_names.json"),
            orient="values",
        )

        for split in ["train", "val", "test"]:
            print(f"-------------{split.upper()}-------------")
            x_arr, y_arr = self._get_split(split)
            np.save(os.path.join(self.args.save_path, f"{split}_x_original.npy"), x_arr)
            np.save(os.path.join(self.args.save_path, f"{split}_y_original.npy"), y_arr)
            print(x_arr.shape, y_arr.shape)


def main():
    parser = argparse.ArgumentParser(description="LG3 extract_forecasting_data_raw")
    parser.add_argument("--input_dir", type=str, default="lg3/data/processed")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--compression_factor", type=int, default=4)
    parser.add_argument("--use_lagged", action="store_true")
    parser.add_argument("--lag_days", type=str, default="0,7,14")
    parser.add_argument("--lag_window_days", type=str, default="3,1,1")
    parser.add_argument("--steps_per_day", type=int, default=24)
    parser.add_argument("--sep_steps", type=int, default=4)
    args = parser.parse_args()

    exp = ExtractRawData(args)
    exp.extract_data()


if __name__ == "__main__":
    main()
