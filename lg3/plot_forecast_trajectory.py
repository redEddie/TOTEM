import argparse
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lg3.lib.models.revin import RevIN


def load_feature_index(processed_data_dir, feature_name):
    """
    Finds feature columns from the processed per-unit data files.
    """
    unit_data_path = os.path.join(processed_data_dir, "smartcare_units")
    sample_files = glob.glob(os.path.join(unit_data_path, "unit_*/lg3_train.csv"))

    # Fallback for old structure if per-unit is not found
    if not sample_files:
        fallback_path = os.path.join(processed_data_dir, "lg3_train.csv")
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                f"No train data found in {unit_data_path} or at {fallback_path} to determine features."
            )
        sample_files = [fallback_path]

    # Read the first file to get column names
    df = pd.read_csv(sample_files[0], nrows=1)
    cols = [c for c in df.columns if c != "Timestamp"]
    if feature_name not in cols:
        raise ValueError(
            f"Feature '{feature_name}' not found in {sample_files[0]}. Available: {cols[:10]}"
        )
    return cols.index(feature_name), cols


def inference(
    x,
    y,
    codeids_x,
    model_decode,
    model_mustd,
    codebook,
    compression,
    device,
    onehot=False,
    scheme=1,
):
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    B, TCin, Sin = codeids_x.shape
    Tout = y.shape[1]

    _ = model_mustd.revin_in(x, "norm")
    _ = model_mustd.revin_out(y.to(device), "norm")

    code_ids = codeids_x.flatten()
    if onehot:
        xcodes = torch.nn.functional.one_hot(code_ids, num_classes=codebook.shape[0])
    else:
        xcodes = codebook[code_ids]
    xcodes = xcodes.reshape((B, TCin, Sin, xcodes.shape[-1]))
    xcodes = torch.permute(xcodes, (0, 2, 1, 3))
    xcodes = xcodes.reshape((B * Sin, TCin, xcodes.shape[-1]))
    xcodes = torch.permute(xcodes, (1, 0, 2))

    ytime = model_decode(xcodes)
    ytime = ytime.reshape((B, Sin, Tout))
    ytime = torch.permute(ytime, (0, 2, 1))

    times = torch.permute(x, (0, 2, 1))
    times = times.reshape((-1, times.shape[-1]))
    ymeanstd = model_mustd(times)
    ymeanstd = ymeanstd.reshape((B, Sin, 2))
    ymeanstd = torch.permute(ymeanstd, (0, 2, 1))
    ymean = ymeanstd[:, 0, :].unsqueeze(1)
    ystd = ymeanstd[:, 1, :].unsqueeze(1)

    if scheme == 1:
        ymean = ymean + model_mustd.revin_in.mean
        ystd = ystd + model_mustd.revin_in.stdev
        ytime = ytime * ystd + ymean
    elif scheme == 2:
        ytime = model_mustd.revin_in(ytime, "denorm")
    else:
        raise ValueError("Unknown prediction scheme %d" % scheme)

    return ytime.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot LG3 power forecast trajectory")
    parser.add_argument("--data_dir", type=str, default="lg3/data/forecasting/Tin96_Tout96")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="lg3/data/processed",
        help="Directory containing processed data, possibly in 'smartcare_units' subfolders.",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_plots", type=int, default=1, help="Number of plots to generate.")
    parser.add_argument("--show_input", type=int, default=50)
    parser.add_argument("--show_pred", type=int, default=50)
    parser.add_argument("--compression", type=int, default=4)
    parser.add_argument("--scheme", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--output", type=str, default="lg3/results/power_trajectory.png")
    args = parser.parse_args()

    feature_idx, cols = load_feature_index(args.processed_data_dir, args.feature)

    x = np.load(os.path.join(args.data_dir, "test_x_original.npy"))
    y = np.load(os.path.join(args.data_dir, "test_y_original.npy"))
    codeids_x = np.load(os.path.join(args.data_dir, "test_x_codes.npy"))
    codebook = np.load(os.path.join(args.data_dir, "codebook.npy"))

    device = "cuda:%d" % args.cuda_id if torch.cuda.is_available() else "cpu"
    model_decode = torch.load(
        os.path.join(args.checkpoint_dir, "decode_checkpoint.pth"),
        map_location=device,
        weights_only=False,
    )
    model_mustd = torch.load(
        os.path.join(args.checkpoint_dir, "mustd_checkpoint.pth"),
        map_location=device,
        weights_only=False,
    )
    model_decode.to(device).eval()
    model_mustd.to(device).eval()

    codebook_t = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    
    output_dir = os.path.dirname(args.output)
    output_basename = os.path.basename(args.output)
    output_filename, output_ext = os.path.splitext(output_basename)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.num_plots):
        current_sample_idx = args.sample_idx + i
        if current_sample_idx >= len(x):
            print(f"Warning: Sample index {current_sample_idx} is out of bounds. Stopping.")
            break
            
        x_sample = torch.from_numpy(x[current_sample_idx : current_sample_idx + 1]).float()
        y_sample = torch.from_numpy(y[current_sample_idx : current_sample_idx + 1]).float()
        code_x_sample = torch.from_numpy(codeids_x[current_sample_idx : current_sample_idx + 1]).long()

        y_pred = inference(
            x_sample,
            y_sample,
            code_x_sample,
            model_decode,
            model_mustd,
            codebook_t,
            args.compression,
            device,
            scheme=args.scheme,
        )

        x_series = x_sample[0, :, feature_idx].numpy()
        y_series = y_sample[0, :, feature_idx].numpy()
        y_pred_series = y_pred[0, :, feature_idx]

        input_len = min(args.show_input, len(x_series))
        pred_len = min(args.show_pred, len(y_series))

        x_axis_input = np.arange(0, input_len)
        x_axis_pred = np.arange(input_len, input_len + pred_len)

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis_input, x_series[:input_len], color="gray", label="input")
        plt.plot(x_axis_pred, y_pred_series[:pred_len], color="blue", label="prediction")
        plt.plot(x_axis_pred, y_series[:pred_len], color="green", label="groundtruth")
        plt.legend()
        plt.title(f"Feature: {args.feature} (Sample: {current_sample_idx})")
        plt.tight_layout()
        
        if args.num_plots > 1:
            current_output_path = os.path.join(output_dir, f"{output_filename}_sample_{current_sample_idx}{output_ext}")
        else:
            current_output_path = args.output

        plt.savefig(current_output_path, dpi=150)
        print("Saved plot to", current_output_path)
        plt.close() # Close the figure to free memory


if __name__ == "__main__":
    main()
