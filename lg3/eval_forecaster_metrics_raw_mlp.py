import argparse
import os

import numpy as np
import pandas as pd
import torch

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
    parser = argparse.ArgumentParser(description="Evaluate raw MLP forecaster and export per-sequence metrics")
    parser.add_argument("--data_dir", type=str, default="lg3/data/forecasting_raw/Tin288_Tout96")
    parser.add_argument("--feature_names_path", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="lg3/results/forecast_metrics_raw_mlp.csv")
    parser.add_argument("--output_seq_mse_csv", type=str, default="lg3/results/forecast_seq_mse_raw_mlp.csv")
    parser.add_argument("--output_seq_mae_csv", type=str, default="lg3/results/forecast_seq_mae_raw_mlp.csv")
    parser.add_argument("--output_seq_mape_csv", type=str, default="lg3/results/forecast_seq_mape_raw_mlp.csv")
    parser.add_argument("--output_seq_mase_csv", type=str, default="lg3/results/forecast_seq_mase_raw_mlp.csv")
    args = parser.parse_args()

    x = np.load(os.path.join(args.data_dir, "test_x_original.npy"))
    y = np.load(os.path.join(args.data_dir, "test_y_original.npy"))

    device = "cuda:%d" % args.cuda_id if torch.cuda.is_available() else "cpu"
    model_decode = torch.load(
        os.path.join(args.checkpoint_dir, "decode_checkpoint.pth"),
        map_location=device,
        weights_only=False,
    )
    model_decode.to(device).eval()

    sin = x.shape[-1]
    revin = RevIN(num_features=sin, affine=False)
    revin.to(device)

    feature_names_path = args.feature_names_path
    if not feature_names_path:
        feature_names_path = os.path.join(args.data_dir, "feature_names.json")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(
            f"feature_names.json not found at {feature_names_path}. "
            "Re-run extract_forecasting_data_raw to generate it."
        )
    feature_names = pd.read_json(feature_names_path, typ="series").tolist()

    if y.shape[-1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: data has {y.shape[-1]} but processed_csv has {len(feature_names)}"
        )

    total_sse = np.zeros(y.shape[-1], dtype=np.float64)
    total_sae = np.zeros(y.shape[-1], dtype=np.float64)
    total_count = 0
    total_ape = np.zeros(y.shape[-1], dtype=np.float64)
    total_ape_count = np.zeros(y.shape[-1], dtype=np.float64)
    total_naive_ae = np.zeros(y.shape[-1], dtype=np.float64)
    total_naive_count = 0
    seq_mse = []
    seq_mae = []
    seq_mape = []
    seq_idx = []
    eps = 1e-6

    with torch.no_grad():
        for i in range(0, len(x), args.batch_size):
            x_batch = torch.from_numpy(x[i : i + args.batch_size]).float()
            y_batch = torch.from_numpy(y[i : i + args.batch_size]).float()

            y_pred = inference_batch(x_batch, y_batch, model_decode, revin, device)

            y_true = y_batch.numpy()
            err = y_pred - y_true
            total_sse += (err ** 2).sum(axis=(0, 1))
            total_sae += np.abs(err).sum(axis=(0, 1))
            total_count += err.shape[0] * err.shape[1]
            denom = np.maximum(np.abs(y_true), eps)
            total_ape += (np.abs(err) / denom).sum(axis=(0, 1))
            total_ape_count += (denom > eps).sum(axis=(0, 1))
            if y_true.shape[1] > 1:
                naive_err = y_true[:, 1:, :] - y_true[:, :-1, :]
                total_naive_ae += np.abs(naive_err).sum(axis=(0, 1))
                total_naive_count += naive_err.shape[0] * naive_err.shape[1]

            per_seq_mse_feat = np.mean(err ** 2, axis=1)
            per_seq_mae_feat = np.mean(np.abs(err), axis=1)
            per_seq_mape_feat = np.mean(np.abs(err) / denom, axis=1)
            per_seq_mse_all = per_seq_mse_feat.mean(axis=1, keepdims=True)
            per_seq_mae_all = per_seq_mae_feat.mean(axis=1, keepdims=True)
            per_seq_mape_all = per_seq_mape_feat.mean(axis=1, keepdims=True)
            seq_mse.extend(np.concatenate([per_seq_mse_all, per_seq_mse_feat], axis=1).tolist())
            seq_mae.extend(np.concatenate([per_seq_mae_all, per_seq_mae_feat], axis=1).tolist())
            seq_mape.extend(np.concatenate([per_seq_mape_all, per_seq_mape_feat], axis=1).tolist())
            seq_idx.extend(range(i, i + len(per_seq_mse_all)))

    mse_feat = total_sse / total_count
    mae_feat = total_sae / total_count
    mape_feat = total_ape / np.maximum(total_ape_count, 1)
    mase_denom = total_naive_ae / max(total_naive_count, 1)
    mase_feat = mae_feat / np.maximum(mase_denom, eps)

    overall_mse = float(mse_feat.mean())
    overall_mae = float(mae_feat.mean())
    overall_mape = float(mape_feat.mean())
    overall_mase = float(mase_feat.mean())

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mse": mse_feat,
            "mae": mae_feat,
            "mape": mape_feat,
            "mase": mase_feat,
        }
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "feature": "__all__",
                        "mse": overall_mse,
                        "mae": overall_mae,
                        "mape": overall_mape,
                        "mase": overall_mase,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    seq_cols = ["__all__"] + feature_names
    seq_mse_df = pd.DataFrame(seq_mse, columns=seq_cols)
    seq_mse_df.insert(0, "seq_idx", seq_idx)
    seq_mae_df = pd.DataFrame(seq_mae, columns=seq_cols)
    seq_mae_df.insert(0, "seq_idx", seq_idx)
    seq_mape_df = pd.DataFrame(seq_mape, columns=seq_cols)
    seq_mape_df.insert(0, "seq_idx", seq_idx)
    if total_naive_count > 0:
        per_seq_mae_arr = np.array(seq_mae)
        per_seq_mae_all = per_seq_mae_arr[:, :1]
        per_seq_mae_feat = per_seq_mae_arr[:, 1:]
        mase_denom_all = float(mase_denom.mean())
        per_seq_mase_all = per_seq_mae_all / max(mase_denom_all, eps)
        per_seq_mase_feat = per_seq_mae_feat / np.maximum(mase_denom, eps)
        per_seq_mase = np.concatenate([per_seq_mase_all, per_seq_mase_feat], axis=1)
        seq_mase_df = pd.DataFrame(per_seq_mase, columns=seq_cols)
        seq_mase_df.insert(0, "seq_idx", seq_idx)
    else:
        seq_mase_df = pd.DataFrame({"seq_idx": seq_idx})
    seq_mse_df.to_csv(args.output_seq_mse_csv, index=False)
    seq_mae_df.to_csv(args.output_seq_mae_csv, index=False)
    seq_mape_df.to_csv(args.output_seq_mape_csv, index=False)
    if not seq_mase_df.empty:
        seq_mase_df.to_csv(args.output_seq_mase_csv, index=False)

    print("Saved metrics to", args.output_csv)
    print("Saved per-sequence MSE to", args.output_seq_mse_csv)
    print("Saved per-sequence MAE to", args.output_seq_mae_csv)
    print("Saved per-sequence MAPE to", args.output_seq_mape_csv)
    if not seq_mase_df.empty:
        print("Saved per-sequence MASE to", args.output_seq_mase_csv)
    print("Overall MSE:", overall_mse)
    print("Overall MAE:", overall_mae)
    print("Overall MAPE:", overall_mape)
    print("Overall MASE:", overall_mase)


if __name__ == "__main__":
    main()
