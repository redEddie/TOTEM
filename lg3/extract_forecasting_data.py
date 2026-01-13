import argparse
import os

import numpy as np
import pandas as pd
import torch

from lg3.lib.models.revin import RevIN


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


def time2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    t_len = revin_data.shape[2]
    compressed_time = int(t_len / compression_factor)

    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, t_len)
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim, compressed_time)
        code_ids = encoding_indices.view(bs, nvar, compressed_time)

    return codes, code_ids, embedding_weight


def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        one_hot_encodings = torch.zeros(
            int(bs * nvars * compressed_len), num_code_words, device=device
        )
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device), 1)
        quantized = torch.matmul(one_hot_encodings, codebook.to(device)).view(input_shape)
        quantized_swaped = torch.swapaxes(quantized, 1, 2)
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1, 2)
        predictions_original_space = revin_layer(predictions_revin_space, "denorm")

    return predictions_revin_space, predictions_original_space


def save_files(path, data_dict, mode, save_codebook):
    np.save(os.path.join(path, f"{mode}_x_original.npy"), data_dict["x_original_arr"])
    np.save(os.path.join(path, f"{mode}_y_original.npy"), data_dict["y_original_arr"])
    np.save(os.path.join(path, f"{mode}_x_codes.npy"), data_dict["x_code_ids_all_arr"])
    np.save(os.path.join(path, f"{mode}_y_codes.npy"), data_dict["y_code_ids_all_arr"])

    if save_codebook:
        np.save(os.path.join(path, "codebook.npy"), data_dict["codebook"])


class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu"
        enc_in = self.args.enc_in
        if enc_in <= 0:
            sample_path = os.path.join(self.args.input_dir, "lg3_train.csv")
            enc_in = load_split_csv(sample_path).shape[1]
        self.revin_layer_x = RevIN(num_features=enc_in, affine=False, subtract_last=False)
        self.revin_layer_y = RevIN(num_features=enc_in, affine=False, subtract_last=False)

    def _get_split(self, split):
        split_path = os.path.join(self.args.input_dir, f"lg3_{split}.csv")
        df = load_split_csv(split_path)
        values = df.to_numpy(dtype=np.float32)
        x, y = build_sequences(values, self.args.seq_len, self.args.pred_len)
        return x, y

    def one_loop_forecasting(self, x_arr, y_arr, vqvae_model):
        x_original_all = []
        y_original_all = []
        x_code_ids_all = []
        y_code_ids_all = []
        x_reverted_all = []
        y_reverted_all = []

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_arr), torch.from_numpy(y_arr)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False
        )

        for i, (batch_x, batch_y) in enumerate(loader):
            if i == 0:
                if batch_x.shape[-1] == batch_y.shape[-1]:
                    num_sensors = batch_x.shape[-1]
                else:
                    raise ValueError("X/Y feature mismatch.")

            x_original_all.append(batch_x.numpy())
            y_original_all.append(batch_y.numpy())

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            x_in_revin_space = self.revin_layer_x(batch_x, "norm")
            y_in_revin_space = self.revin_layer_y(batch_y, "norm")

            x_codes, x_code_ids, codebook = time2codes(
                x_in_revin_space.permute(0, 2, 1),
                self.args.compression_factor,
                vqvae_model.encoder,
                vqvae_model.vq,
            )
            y_codes, y_code_ids, codebook = time2codes(
                y_in_revin_space.permute(0, 2, 1),
                self.args.compression_factor,
                vqvae_model.encoder,
                vqvae_model.vq,
            )

            x_code_ids_all.append(x_code_ids.detach().cpu().numpy())
            y_code_ids_all.append(y_code_ids.detach().cpu().numpy())

            x_predictions_revin_space, x_predictions_original_space = codes2time(
                x_code_ids,
                codebook,
                self.args.compression_factor,
                vqvae_model.decoder,
                self.revin_layer_x,
            )
            y_predictions_revin_space, y_predictions_original_space = codes2time(
                y_code_ids,
                codebook,
                self.args.compression_factor,
                vqvae_model.decoder,
                self.revin_layer_y,
            )

            x_reverted_all.append(x_predictions_original_space.detach().cpu().numpy())
            y_reverted_all.append(y_predictions_original_space.detach().cpu().numpy())

        x_original_arr = np.concatenate(x_original_all, axis=0)
        y_original_arr = np.concatenate(y_original_all, axis=0)

        x_code_ids_all_arr = np.concatenate(x_code_ids_all, axis=0)
        y_code_ids_all_arr = np.concatenate(y_code_ids_all, axis=0)

        x_reverted_all_arr = np.concatenate(x_reverted_all, axis=0)
        y_reverted_all_arr = np.concatenate(y_reverted_all, axis=0)

        data_dict = {}
        data_dict["x_original_arr"] = x_original_arr
        data_dict["y_original_arr"] = y_original_arr
        data_dict["x_code_ids_all_arr"] = np.swapaxes(x_code_ids_all_arr, 1, 2)
        data_dict["y_code_ids_all_arr"] = np.swapaxes(y_code_ids_all_arr, 1, 2)
        data_dict["x_reverted_all_arr"] = x_reverted_all_arr
        data_dict["y_reverted_all_arr"] = y_reverted_all_arr
        data_dict["codebook"] = codebook.detach().cpu().numpy()

        if data_dict["x_original_arr"].shape[-1] != num_sensors:
            raise ValueError("Sensor dimension mismatch.")

        print(data_dict["x_original_arr"].shape, data_dict["y_original_arr"].shape)
        print(data_dict["x_code_ids_all_arr"].shape, data_dict["y_code_ids_all_arr"].shape)
        print(data_dict["x_reverted_all_arr"].shape, data_dict["y_reverted_all_arr"].shape)
        print(data_dict["codebook"].shape)

        return data_dict

    def extract_data(self):
        vqvae_model = torch.load(self.args.trained_vqvae_model_path, weights_only=False)
        vqvae_model.to(self.device)
        vqvae_model.eval()

        if self.args.seq_len % self.args.compression_factor != 0:
            raise ValueError("seq_len must be divisible by compression_factor.")
        if self.args.pred_len % self.args.compression_factor != 0:
            raise ValueError("pred_len must be divisible by compression_factor.")

        os.makedirs(self.args.save_path, exist_ok=True)

        print("-------------TRAIN-------------")
        x_train, y_train = self._get_split("train")
        train_data_dict = self.one_loop_forecasting(x_train, y_train, vqvae_model)
        save_files(self.args.save_path, train_data_dict, "train", save_codebook=True)

        print("-------------VAL-------------")
        x_val, y_val = self._get_split("val")
        val_data_dict = self.one_loop_forecasting(x_val, y_val, vqvae_model)
        save_files(self.args.save_path, val_data_dict, "val", save_codebook=False)

        print("-------------TEST-------------")
        x_test, y_test = self._get_split("test")
        test_data_dict = self.one_loop_forecasting(x_test, y_test, vqvae_model)
        save_files(self.args.save_path, test_data_dict, "test", save_codebook=True)


def main():
    parser = argparse.ArgumentParser(description="LG3 extract_forecasting_data")
    parser.add_argument("--input_dir", type=str, default="lg3/data/processed")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--compression_factor", type=int, default=4)
    parser.add_argument("--trained_vqvae_model_path", type=str, required=True)
    args = parser.parse_args()

    exp = ExtractData(args)
    exp.extract_data()


if __name__ == "__main__":
    main()
