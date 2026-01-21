import argparse
import os

import json
import numpy as np
import torch
import matplotlib.pyplot as plt


def inference_batch(
    x,
    y,
    codeids_x,
    model_decode,
    model_mustd,
    codebook,
    device,
    scheme=1,
):
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    bsz, tcin, sin = codeids_x.shape
    tout = y.shape[1]

    _ = model_mustd.revin_in(x, "norm")
    _ = model_mustd.revin_out(y.to(device), "norm")

    code_ids = codeids_x.flatten()
    xcodes = codebook[code_ids]
    xcodes = xcodes.reshape((bsz, tcin, sin, xcodes.shape[-1]))
    xcodes = torch.permute(xcodes, (0, 2, 1, 3))
    xcodes = xcodes.reshape((bsz * sin, tcin, xcodes.shape[-1]))
    xcodes = torch.permute(xcodes, (1, 0, 2))

    ytime = model_decode(xcodes)
    ytime = ytime.reshape((bsz, sin, tout))
    ytime = torch.permute(ytime, (0, 2, 1))

    times = torch.permute(x, (0, 2, 1))
    times = times.reshape((-1, times.shape[-1]))
    ymeanstd = model_mustd(times)
    ymeanstd = ymeanstd.reshape((bsz, sin, 2))
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
        raise ValueError(f"Unknown prediction scheme {scheme}")

    return ytime.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot overlapping forecasts vs ground truth")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--feature_names_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--feature", type=str, default="Power")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_sequences", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--scheme", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="pred_only",
        choices=["pred_only", "context"],
        help="pred_only: plot only forecast horizon. context: plot input tail + forecast.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=0,
        help="Number of input steps to show when plot_mode=context. 0 uses full input length.",
    )
    parser.add_argument("--output", type=str, default="lg3/results/forecast_overlap_power.png")
    parser.add_argument("--save_each", action="store_true", help="Save each sequence as a separate plot.")
    parser.add_argument("--output_dir", type=str, default="lg3/results/non_overlap")
    args = parser.parse_args()

    if args.save_each and os.path.isdir(args.output_dir):
        for name in os.listdir(args.output_dir):
            path = os.path.join(args.output_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    x = np.load(os.path.join(args.data_dir, "test_x_original.npy"))
    y = np.load(os.path.join(args.data_dir, "test_y_original.npy"))
    codeids_x = np.load(os.path.join(args.data_dir, "test_x_codes.npy"))
    codebook = np.load(os.path.join(args.data_dir, "codebook.npy"))

    with open(args.feature_names_path, "r") as f:
        feature_names = json.load(f)
    if isinstance(feature_names, dict):
        feature_names = feature_names.get("features", [])
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
    model_mustd = torch.load(
        os.path.join(args.checkpoint_dir, "mustd_checkpoint.pth"),
        map_location=device,
        weights_only=False,
    )
    model_decode.to(device).eval()
    model_mustd.to(device).eval()

    codebook_t = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)

    preds = []
    gts = []
    x_hist = []
    with torch.no_grad():
        for i in range(start, end, args.batch_size):
            xb = torch.from_numpy(x[i : i + args.batch_size]).float()
            yb = torch.from_numpy(y[i : i + args.batch_size]).float()
            cb = torch.from_numpy(codeids_x[i : i + args.batch_size]).long()
            y_pred = inference_batch(
                xb,
                yb,
                cb,
                model_decode,
                model_mustd,
                codebook_t,
                device,
                scheme=args.scheme,
            )
            preds.append(y_pred[:, :, feature_idx])
            gts.append(yb.numpy()[:, :, feature_idx])
            x_hist.append(xb.numpy()[:, :, feature_idx])

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    x_hist = np.concatenate(x_hist, axis=0)

    plt.figure(figsize=(12, 4))
    for i in range(len(preds)):
        is_individual = args.save_each
        if args.plot_mode == "pred_only":
            offset = start + i
            x_axis = np.arange(offset, offset + preds.shape[1])
            alpha = 1.0 if is_individual else 0.15
            plt.plot(x_axis, preds[i], color="tab:blue", alpha=alpha)
            plt.plot(x_axis, gts[i], color="tab:green", alpha=alpha)
        else:
            full_ctx = x_hist[i]
            ctx_len = args.context_len if args.context_len > 0 else full_ctx.shape[0]
            ctx = full_ctx[-ctx_len:]
            x_axis_ctx = np.arange(0, ctx_len)
            x_axis_pred = np.arange(ctx_len, ctx_len + preds.shape[1])
            alpha = 1.0 if is_individual else 0.2
            plt.plot(x_axis_ctx, ctx, color="tab:gray", alpha=alpha)
            plt.plot(x_axis_pred, preds[i], color="tab:blue", alpha=alpha)
            plt.plot(x_axis_pred, gts[i], color="tab:green", alpha=alpha)
        if args.save_each:
            os.makedirs(args.output_dir, exist_ok=True)
            plt.title(f"{args.feature} seq {start + i}")
            plt.xlabel("time step")
            plt.ylabel(args.feature)
            out = os.path.join(args.output_dir, f"seq_{start + i:05d}.png")
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.cla()

    title = f"Forecast vs ground truth ({args.feature})"
    if args.plot_mode == "context":
        title += " with input context"
    if not args.save_each:
        plt.title(title)
        plt.xlabel("time step")
        plt.ylabel(args.feature)
        plt.tight_layout()

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        plt.savefig(args.output, dpi=150)
        print("Saved overlap plot to", args.output)


if __name__ == "__main__":
    main()
