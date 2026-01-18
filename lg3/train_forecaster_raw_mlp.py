import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lg3.lib.models.decode import SimpleMLP
from lg3.lib.models.metrics import pearsoncor
from lg3.lib.models.revin import RevIN
from lg3.lib.utils.checkpoint import EarlyStopping
from lg3.lib.utils.env import seed_all_rng


def create_dataloader(datapath="/data", batchsize=8):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        timex_file = os.path.join(datapath, f"{split}_x_original.npy")
        timey_file = os.path.join(datapath, f"{split}_y_original.npy")
        timex = torch.from_numpy(np.load(timex_file)).to(dtype=torch.float32)
        timey = torch.from_numpy(np.load(timey_file)).to(dtype=torch.float32)

        print("[Dataset][%s] %d of examples" % (split, timex.shape[0]))

        dataset = torch.utils.data.TensorDataset(timex, timey)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True if split == "train" else False,
            num_workers=10,
            drop_last=True if split == "train" else False,
        )

    return dataloaders


def parse_int_list(value):
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def loss_fn(loss_type, beta=1.0):
    if loss_type == "mse":
        loss = nn.MSELoss()
    elif loss_type == "smoothl1":
        loss = nn.SmoothL1Loss(beta=beta)
    else:
        raise ValueError("Invalid type")
    return loss


def build_mlp_input(x):
    bsz, tin, sin = x.shape
    x_feat = torch.permute(x, (0, 2, 1))
    x_feat = x_feat.reshape(bsz * sin, tin)
    return x_feat, bsz, sin


def train_one_epoch(
    dataloader,
    model_decode,
    revin,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_type: str = "smoothl1",
    beta: float = 1.0,
):
    running_loss = 0.0
    log_every = max(len(dataloader) // 3, 3)

    lossfn = loss_fn(loss_type, beta=beta)
    for i, data in enumerate(dataloader):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        x_norm = revin(x, "norm")
        y_norm = (y - revin.mean) / revin.stdev

        xcodes, bsz, sin = build_mlp_input(x_norm)
        y_pred_norm = model_decode(xcodes)
        tout = y.shape[1]
        y_pred_norm = y_pred_norm.reshape((bsz, sin, tout))
        y_pred_norm = torch.permute(y_pred_norm, (0, 2, 1))

        loss = lossfn(y_pred_norm, y_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_every == log_every - 1:
            last_loss = running_loss / log_every
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
                f"lr {lr:02.5f} | loss {last_loss:5.4f}"
            )
            running_loss = 0.0

        if scheduler is not None:
            scheduler.step()


def inference(data, model_decode, revin, device):
    x, y = data
    x = x.to(device)

    x_norm = revin(x, "norm")
    xcodes, bsz, sin = build_mlp_input(x_norm)
    y_pred_norm = model_decode(xcodes)
    tout = y.shape[1]
    y_pred_norm = y_pred_norm.reshape((bsz, sin, tout))
    y_pred_norm = torch.permute(y_pred_norm, (0, 2, 1))
    y_pred = revin(y_pred_norm, "denorm")
    return y_pred


def train(args):
    if not os.path.exists(args.file_save_path):
        os.makedirs(args.file_save_path)

    save_name = (
        str(args.data_type)
        + "_Tin"
        + str(args.Tin)
        + "_Tout"
        + str(args.Tout)
        + "_seed"
        + str(args.seed)
        + "_raw_mlp.txt"
    )
    save_file = open(args.file_save_path + save_name, "w+")

    device = torch.device("cuda:%d" % (args.cuda_id))
    torch.cuda.set_device(device)

    seed_all_rng(None if args.seed < 0 else args.seed)

    dataloaders = create_dataloader(datapath=args.data_path, batchsize=args.batchsize)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    sample = np.load(os.path.join(args.data_path, "train_x_original.npy"))
    sin = sample.shape[-1]

    mlp_hidden_dims = parse_int_list(args.mlp_hidden_dims)

    model_decode = SimpleMLP(
        in_dim=args.Tin,
        out_dim=args.Tout,
        hidden_dims=mlp_hidden_dims,
        dropout=args.mlp_dropout,
    )
    revin = RevIN(num_features=sin, affine=False)

    model_decode.to(device)
    revin.to(device)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model_decode.parameters(), lr=args.baselr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_decode.parameters(), lr=args.baselr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model_decode.parameters(), lr=args.baselr)
    else:
        raise ValueError("Uknown optimizer type %s" % (args.optimizer))

    if args.scheduler == "step":
        step_lr_in_iters = args.steps * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_lr_in_iters, gamma=0.1)
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.baselr,
            steps_per_epoch=len(train_dataloader),
            epochs=args.epochs,
            pct_start=0.2,
        )
    else:
        raise ValueError("Uknown scheduler type %s" % (args.scheduler))

    for epoch in range(args.epochs):
        model_decode.train()
        train_one_epoch(
            train_dataloader,
            model_decode,
            revin,
            optimizer,
            scheduler,
            epoch,
            device,
            beta=args.beta,
        )

        if val_dataloader is not None:
            model_decode.eval()
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for vdata in val_dataloader:
                    pred_time = inference(vdata, model_decode, revin, device)
                    labels_time = vdata[1].to(device)

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]
            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            print(
                f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}"
            )

            save_file.write(
                f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}\n"
            )

            early_stopping_counter = early_stopping(
                running_mse, running_mae, {"decode": model_decode}
            )

        if test_dataloader is not None:
            model_decode.eval()
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for tdata in test_dataloader:
                    pred_time = inference(tdata, model_decode, revin, device)
                    labels_time = tdata[1].to(device)

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]

            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            print(
                f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}"
            )
            save_file.write(
                f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}\n"
            )
            save_file.write(f"Early stopping counter is: {early_stopping_counter}\n")

        if early_stopping.early_stop:
            print("Early stopping....")
            save_file.write("Early stopping....")
            save_file.write("Take the test values right before the last early stopping counter = 0")
            save_file.close()
            return


def default_argument_parser():
    parser = argparse.ArgumentParser(description="LG3 Raw Forecaster (MLP)")
    parser.add_argument("--cuda-id", default=0, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--data-type", default="lg3", type=str)
    parser.add_argument("--Tin", default=128, type=int)
    parser.add_argument("--Tout", default=24, type=int)
    parser.add_argument("--data_path", default="", type=str)

    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", default="/data/", type=str)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--scheduler", default="onecycle", type=str)
    parser.add_argument("--baselr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--steps", default=4, type=int)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mlp_hidden_dims", default="1024,1024,1024", type=str)
    parser.add_argument("--mlp_dropout", default=0.1, type=float)
    parser.add_argument("--file_save_path", default="", type=str)
    parser.add_argument("--batchsize", default=64, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    train(args)
