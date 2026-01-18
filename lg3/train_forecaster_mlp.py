import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lg3.lib.models.decode import MuStdModel, SimpleMLP
from lg3.lib.models.metrics import pearsoncor
from lg3.lib.models.revin import RevIN
from lg3.lib.utils.checkpoint import EarlyStopping
from lg3.lib.utils.env import seed_all_rng


def create_time_series_dataloader(datapath="/data", batchsize=8):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        timex_file = os.path.join(datapath, "%s_x_original.npy" % split)
        timex = np.load(timex_file)
        timex = torch.from_numpy(timex).to(dtype=torch.float32)

        timey_file = os.path.join(datapath, "%s_y_original.npy" % split)
        timey = np.load(timey_file)
        timey = torch.from_numpy(timey).to(dtype=torch.float32)

        codex_file = os.path.join(datapath, "%s_x_codes.npy" % (split))
        codex = np.load(codex_file)
        codex = torch.from_numpy(codex).to(dtype=torch.int64)

        codey_oracle_file = os.path.join(datapath, "%s_y_codes_oracle.npy" % split)
        if not os.path.exists(codey_oracle_file):
            codey_oracle_file = os.path.join(datapath, "%s_y_codes.npy" % split)
        codey_oracle = np.load(codey_oracle_file)
        codey_oracle = torch.from_numpy(codey_oracle).to(dtype=torch.int64)

        print("[Dataset][%s] %d of examples" % (split, timex.shape[0]))

        dataset = torch.utils.data.TensorDataset(timex, timey, codex, codey_oracle)
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


def build_mlp_input(codeids_x, codebook, onehot=False):
    bsz, tcin, sin = codeids_x.shape
    code_ids = codeids_x.flatten()
    if onehot:
        xcodes = F.one_hot(code_ids, num_classes=codebook.shape[0]).float()
    else:
        xcodes = codebook[code_ids]
    xcodes = xcodes.reshape((bsz, tcin, sin, xcodes.shape[-1]))
    xcodes = torch.permute(xcodes, (0, 2, 1, 3))
    xcodes = xcodes.reshape((bsz * sin, -1))
    return xcodes, bsz, sin


def train_one_epoch(
    dataloader,
    model_decode,
    model_mustd,
    codebook,
    compression,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_type: str = "smoothl1",
    beta: float = 1.0,
    onehot: bool = False,
    scheme: int = 2,
):
    running_loss, last_loss = 0.0, 0.0
    running_loss_mu, last_loss_mu = 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)

    lossfn = loss_fn(loss_type, beta=beta)
    for i, data in enumerate(dataloader):
        x, y, codeids_x, codeids_y_labels = data
        x = x.to(device)
        y = y.to(device)
        codeids_x = codeids_x.to(device)
        codeids_y_labels = codeids_y_labels.to(device)

        _ = model_mustd.revin_in(x, "norm")
        norm_y = model_mustd.revin_out(y, "norm")

        _, tcout, sout = codeids_y_labels.shape
        tout = tcout * compression
        assert tout == y.shape[1], "%d" % (tcout)

        xcodes, bsz, sin = build_mlp_input(codeids_x, codebook, onehot=onehot)
        ytime = model_decode(xcodes)
        ytime = ytime.reshape((bsz, sin, tout))
        ytime = torch.permute(ytime, (0, 2, 1))

        times = torch.permute(x, (0, 2, 1))
        times = times.reshape((-1, times.shape[-1]))
        ymeanstd = model_mustd(times)

        ymeanstd = ymeanstd.reshape((bsz, sout, 2))
        ymeanstd = torch.permute(ymeanstd, (0, 2, 1))
        ymean = ymeanstd[:, 0, :].unsqueeze(1)
        ystd = ymeanstd[:, 1, :].unsqueeze(1)

        if scheme == 1:
            loss_mu = lossfn(model_mustd.revin_out.mean - model_mustd.revin_in.mean, ymean)
            loss_std = lossfn(model_mustd.revin_out.stdev - model_mustd.revin_in.stdev, ystd)
            loss_decode = lossfn(ytime, norm_y)
            loss_all = lossfn(
                ytime * (ystd.detach() + model_mustd.revin_in.stdev)
                + (ymean.detach() + model_mustd.revin_in.mean),
                y,
            )

            loss = loss_decode + loss_mu + loss_std + loss_all
        elif scheme == 2:
            ytime = model_mustd.revin_in(ytime, "denorm")
            loss_decode = lossfn(ytime, y)
            loss_mu = loss_std = torch.zeros((1,), device=device)
            loss = loss_decode
        else:
            raise ValueError("Unknown prediction scheme %d" % scheme)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_mu += loss_mu.item()
        if i % log_every == log_every - 1:
            last_loss = running_loss / log_every
            last_loss_mu = running_loss_mu / log_every
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
                f"lr {lr:02.5f} | loss {last_loss:5.4f} | loss_mu {last_loss_mu:5.4f}"
            )
            running_loss = 0.0
            running_loss_mu = 0.0

        if scheduler is not None:
            scheduler.step()


def inference(
    data,
    model_decode,
    model_mustd,
    codebook,
    compression,
    device,
    onehot: bool = False,
    scheme: int = 2,
):
    x, y, codeids_x, codeids_y_labels = data
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    _, tcout, sout = codeids_y_labels.shape
    tout = tcout * compression
    del codeids_y_labels

    _ = model_mustd.revin_in(x, "norm")
    _ = model_mustd.revin_out(y, "norm")

    xcodes, bsz, sin = build_mlp_input(codeids_x, codebook, onehot=onehot)
    ytime = model_decode(xcodes)
    ytime = ytime.reshape((bsz, sin, tout))
    ytime = torch.permute(ytime, (0, 2, 1))

    times = torch.permute(x, (0, 2, 1))
    times = times.reshape((-1, times.shape[-1]))
    ymeanstd = model_mustd(times)

    ymeanstd = ymeanstd.reshape((bsz, sout, 2))
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

    return ytime


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
        + "_mlp.txt"
    )
    save_file = open(args.file_save_path + save_name, "w+")

    device = torch.device("cuda:%d" % (args.cuda_id))
    torch.cuda.set_device(device)

    seed_all_rng(None if args.seed < 0 else args.seed)

    datapath = args.data_path
    compression = args.compression
    batchsize = args.batchsize

    sample = np.load(os.path.join(datapath, "train_x_original.npy"))
    sin = sample.shape[-1]
    sout = np.load(os.path.join(datapath, "train_y_original.npy")).shape[-1]

    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, vocab_dim = codebook.shape

    assert vocab_size == args.codebook_size
    dim = vocab_size if args.onehot else vocab_dim

    dataloaders = create_time_series_dataloader(datapath=datapath, batchsize=batchsize)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    mlp_hidden_dims = parse_int_list(args.mlp_hidden_dims)
    mlp_in_dim = (args.Tin // compression) * dim

    model_decode = SimpleMLP(
        in_dim=mlp_in_dim,
        out_dim=args.Tout,
        hidden_dims=mlp_hidden_dims,
        dropout=args.mlp_dropout,
    )

    model_mustd = MuStdModel(
        Tin=args.Tin,
        Tout=args.Tout,
        hidden_dims=[512, 512],
        dropout=0.2,
        is_mlp=True,
    )
    model_mustd.revin_in = RevIN(num_features=sin, affine=False)
    model_mustd.revin_out = RevIN(num_features=sout, affine=False)

    model_decode.to(device)
    model_mustd.to(device)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    model_params = list(model_decode.parameters()) + list(model_mustd.parameters())
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=args.baselr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_params, lr=args.baselr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model_params, lr=args.baselr)
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
        model_mustd.train()
        train_one_epoch(
            train_dataloader,
            model_decode,
            model_mustd,
            codebook,
            args.compression,
            optimizer,
            scheduler,
            epoch,
            device,
            beta=args.beta,
            onehot=args.onehot,
            scheme=args.scheme,
        )

        if val_dataloader is not None:
            model_decode.eval()
            model_mustd.eval()
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for vdata in val_dataloader:
                    pred_time = inference(
                        vdata,
                        model_decode,
                        model_mustd,
                        codebook,
                        args.compression,
                        device,
                        onehot=args.onehot,
                        scheme=args.scheme,
                    )
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
                running_mse, running_mae, {"decode": model_decode, "mustd": model_mustd}
            )

        if test_dataloader is not None:
            model_decode.eval()
            model_mustd.eval()
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for tdata in test_dataloader:
                    pred_time = inference(
                        tdata,
                        model_decode,
                        model_mustd,
                        codebook,
                        args.compression,
                        device,
                        onehot=args.onehot,
                        scheme=args.scheme,
                    )
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
    parser = argparse.ArgumentParser(description="LG3 Code Prediction (MLP)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-id", default=0, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--data-type", default="lg3", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--Tin", default=96, type=int)
    parser.add_argument("--Tout", default=96, type=int)
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
    parser.add_argument("--onehot", action="store_true")
    parser.add_argument("--scheme", default=1, type=int)
    parser.add_argument("--mlp_hidden_dims", default="1024,1024,1024", type=str)
    parser.add_argument("--mlp_dropout", default=0.1, type=float)
    parser.add_argument("--file_save_path", default="", type=str)
    parser.add_argument("--batchsize", default=64, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    train(args)
