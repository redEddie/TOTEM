import argparse
import os

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from lg3.lib.models.revin import RevIN


def build_mlp_input(x):
    bsz, tin, sin = x.shape
    x_feat = torch.permute(x, (0, 2, 1))
    x_feat = x_feat.reshape(bsz * sin, tin)
    return x_feat, bsz, sin


def inference_batch(x, y, model_decode, revin, device):
    x = x.to(device)
    x_norm = revin(x, "norm")
    xcodes, bsz, sin = build_mlp_input(x_norm)

    y_pred_norm = model_decode(xcodes)
    tout = y.shape[1]
    y_pred_norm = y_pred_norm.reshape((bsz, sin, tout))
    y_pred_norm = torch.permute(y_pred_norm, (0, 2, 1))
    y_pred = revin(y_pred_norm, "denorm")
    return y_pred.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot overlapping raw MLP forecasts vs ground truth")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--feature_names_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--feature", type=str, default="Power")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_sequences", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--output", type=str, default="lg3/results/forecast_overlap_power_raw_mlp.png")
    args = parser.parse_args()

    x = np.load(os.path.join(args.data_dir, "test_x_original.npy"))
    y = np.load(os.path.join(args.data_dir, "test_y_original.npy"))

    with open(args.feature_names_path, "r") as f:
        feature_names = json.load(f)
    if args.feature not in feature_names:
        raise ValueError(f"Feature '{args.feature}' not in feature_names.json")
    feature_idx = feature_names.index(args.feature)

    start = max(args.start_idx, 0)
    end = min(start + args.num_sequences, len(x))
    if end <= start:
        raise ValueError("No sequences selected for plotting.")

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    model_decode = torch.load(
        os.path.join(args.checkpoint_dir, "decode_checkpoint.pth"),
        map_location=device,
        weights_only=False,
    )
    model_decode.to(device).eval()

    sin = x.shape[-1]
    revin = RevIN(num_features=sin, affine=False)
    revin.to(device)

    preds = []
    gts = []
    with torch.no_grad():
        for i in range(start, end, args.batch_size):
            xb = torch.from_numpy(x[i : i + args.batch_size]).float()
            yb = torch.from_numpy(y[i : i + args.batch_size]).float()
            y_pred = inference_batch(xb, yb, model_decode, revin, device)
            preds.append(y_pred[:, :, feature_idx])
            gts.append(yb.numpy()[:, :, feature_idx])

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    plt.figure(figsize=(12, 4))
    for i in range(len(preds)):
        offset = start + i
        x_axis = np.arange(offset, offset + preds.shape[1])
        plt.plot(x_axis, preds[i], color="tab:blue", alpha=0.15)
        plt.plot(x_axis, gts[i], color="tab:green", alpha=0.15)

    plt.title(f"Overlap forecast vs ground truth ({args.feature})")
    plt.xlabel("sequence offset")
    plt.ylabel(args.feature)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print("Saved overlap plot to", args.output)


if __name__ == "__main__":
    main()
