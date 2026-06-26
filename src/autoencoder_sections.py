"""Autoencoder helpers for the SPX-only AERGARCH notebook example.

This module preserves the original autoencoder training and rolling-window
logic, while wrapping it for the SPX GitHub example and the 12-12-1 structure.
"""

from pathlib import Path
import copy
import pickle
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .aergarch_functions import rgarch_lg_dr_fit_pred_process
from .model_sections import RESULTS_DIR, collect_single_measure_results


DEFAULT_AE_ARCHITECTURE_12121 = [12, 12, 1, 12, 12]
DEFAULT_AE_HPARAMS = {
    "FINETUNE_EPOCHS": 200,
    "FINETUNE_LR": 0.1,
    "LAMDA_L2": 0.1,
    "PATIENCE": 10,
    "MIN_DELTA": 1e-2,
    "BETA_LIST": [1.0, 1.0],
}

if torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None

__all__ = [
    "DEFAULT_AE_ARCHITECTURE_12121",
    "DEFAULT_AE_HPARAMS",
    "device",
    "SingleLayerAE",
    "StackedAutoencoder",
    "pretrain_layer",
    "fine_tune",
    "rolling_ae_for_market",
    "run_rolling_ae_spx",
    "run_ae_realgarch_spx",
    "collect_ae_results",
]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for the Autoencoder section. "
            "Run this section in the notebook kernel/environment that has torch installed."
        )


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


if torch is not None:

    class SingleLayerAE(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.encoder = nn.Linear(input_size, hidden_size)
            self.decoder = nn.Linear(hidden_size, input_size)

        def encode(self, x):
            return torch.sigmoid(self.encoder(x))

        def decode(self, encoded):
            return torch.sigmoid(self.decoder(encoded))

        def forward(self, x):
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded


    class StackedAutoencoder(nn.Module):
        def __init__(self, layer_sizes):
            super().__init__()
            self.layer_sizes = layer_sizes
            self.bottleneck_index = np.argmin(layer_sizes)

            self.encoder_layers = nn.ModuleList()
            for i in range(self.bottleneck_index):
                self.encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            self.decoder_layers = nn.ModuleList()
            for i in range(self.bottleneck_index, len(layer_sizes) - 1):
                self.decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        def forward(self, x):
            encoder_activations = []

            for lyr in self.encoder_layers:
                x = torch.sigmoid(lyr(x))
                encoder_activations.append(x)

            for lyr in self.decoder_layers:
                x = torch.sigmoid(lyr(x))

            return x, encoder_activations

else:

    class SingleLayerAE:
        def __init__(self, *args, **kwargs):
            _require_torch()


    class StackedAutoencoder:
        def __init__(self, *args, **kwargs):
            _require_torch()


def pretrain_layer(model, data_tensor, epochs, learning_rate, min_delta, patience):
    _require_torch()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    wait = 0

    data_tensor = data_tensor.to(device)
    loss_history = []
    for epoch in range(epochs):
        x_hat = model(data_tensor)
        loss = criterion(x_hat, data_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = loss.item()
        loss_history.append(avg_loss)

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch + 1}, loss={avg_loss:.6f}")
                break

    return best_loss, loss_history


def fine_tune(
    model,
    data_tensor,
    num_epochs,
    learning_rate,
    lamda,
    rho_list,
    beta_list,
    min_delta,
    patience,
):
    _require_torch()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_model_state = None
    wait = 0

    rho_list = [torch.tensor(r, device=device) for r in rho_list]
    beta_list = [torch.tensor(b, device=device) for b in beta_list]

    data_tensor = data_tensor.to(device)

    total_loss_history = []
    mse_loss_history = []
    sparsity_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        outputs, encoder_activations = model(data_tensor)
        mse_loss = criterion(outputs, data_tensor)

        sparsity_loss = 0
        for i, rho in enumerate(rho_list):
            rho_hat = torch.mean(encoder_activations[i], dim=0)
            rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
            kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
                (1 - rho) / (1 - rho_hat)
            )
            sparsity_loss += beta_list[i] * torch.sum(kl)

        total_loss = mse_loss + sparsity_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item()
        avg_mse = mse_loss.item()
        avg_sparse = sparsity_loss.item()

        total_loss_history.append(avg_loss)
        mse_loss_history.append(avg_mse)
        sparsity_loss_history.append(avg_sparse)

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            wait = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        outputs, encoder_activations = model(data_tensor)
        final_mse_loss = criterion(outputs, data_tensor)

        final_sparsity_loss = 0
        for i, rho in enumerate(rho_list):
            rho_hat = torch.mean(encoder_activations[i], dim=0)
            rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
            kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
                (1 - rho) / (1 - rho_hat)
            )
            final_sparsity_loss += beta_list[i] * torch.sum(kl)

        final_total_loss = final_mse_loss + final_sparsity_loss

    final_total_item = final_total_loss.item()

    return (
        model,
        final_total_item,
        encoder_activations,
        total_loss_history,
        mse_loss_history,
        sparsity_loss_history,
    )


def rolling_ae_for_market(market_name, rms_insample_dict, rms_outsample_dict, architecture, hparams):
    _require_torch()
    market_insample_df = rms_insample_dict[market_name]
    market_outsample_df = rms_outsample_dict[market_name]

    FINETUNE_EPOCHS = hparams["FINETUNE_EPOCHS"]
    FINETUNE_LR = hparams["FINETUNE_LR"]
    LAMDA_L2 = hparams["LAMDA_L2"]
    PATIENCE = hparams["PATIENCE"]
    MIN_DELTA = hparams["MIN_DELTA"]
    BETA_LIST = hparams["BETA_LIST"]

    n_insample = market_insample_df.shape[0]
    n_outsample = market_outsample_df.shape[0]
    
    encoded_series_matrix = np.zeros((n_insample, n_outsample + 1))

    weights_history = []
    start_time = time.time()

    for t in range(n_outsample + 1):
        data_df_np = np.vstack([market_insample_df.iloc[t:, :], market_outsample_df.iloc[:t, :]])

        scaler = MinMaxScaler()
        data_normalized_np = scaler.fit_transform(data_df_np)
        data_tensor = torch.tensor(data_normalized_np, dtype=torch.float32)

        mean_value = data_tensor.mean().item()
        num_encoder_layers = np.argmin(architecture)
        RHO_LIST = [mean_value] * num_encoder_layers

        stacked_model = StackedAutoencoder(architecture).to(device)

        trained_model, best_loss, encoder_activations_list, _, _, _ = fine_tune(
            stacked_model,
            data_tensor,
            FINETUNE_EPOCHS,
            FINETUNE_LR,
            LAMDA_L2,
            RHO_LIST,
            BETA_LIST,
            MIN_DELTA,
            PATIENCE,
        )

        if t % 50 == 0 or t == n_outsample:
            print(f"  {market_name} [Step {t}/{n_outsample}] Best Loss: {best_loss:.6f}")

        current_step_layers = []
        for layer in trained_model.encoder_layers:
            w = layer.weight.detach().cpu().numpy().copy()
            current_step_layers.append(w)

        weights_history.append(current_step_layers)

        trained_model.eval()
        compressed_data = encoder_activations_list[-1]
        compressed_data_np = compressed_data.cpu().numpy()

        scaled_compressed = (
            (compressed_data_np - compressed_data_np.min())
            / (compressed_data_np.max() - compressed_data_np.min())
            * (data_df_np.max() - data_df_np.min())
            + data_df_np.min()
        )

        encoded_series_matrix[:, t] = scaled_compressed.flatten()

    end_time = time.time()
    print(f"--- Finished market: {market_name}. Time taken: {end_time - start_time:.2f}s ---")

    result_df = pd.DataFrame(encoded_series_matrix)
    return result_df, weights_history


def run_rolling_ae_spx(
    market_list,
    rms_insample,
    rms_outsample,
    architecture=None,
    hparams=None,
    use_cache=True,
    overwrite=False,
    save=True,
):
    architecture = DEFAULT_AE_ARCHITECTURE_12121 if architecture is None else architecture
    hparams = DEFAULT_AE_HPARAMS.copy() if hparams is None else hparams

    encoded_path = RESULTS_DIR / "ae_encoded_series_12121_spx.pkl"
    weights_path = RESULTS_DIR / "ae_encoder_weights_12121_spx.pkl"

    if use_cache and encoded_path.exists() and weights_path.exists() and not overwrite:
        return _load_pickle(encoded_path), _load_pickle(weights_path)

    all_market_encoded_series = {}
    all_market_weights = {}

    for market in market_list:
        market_result_df, market_weights = rolling_ae_for_market(
            market_name=market,
            rms_insample_dict=rms_insample,
            rms_outsample_dict=rms_outsample,
            architecture=architecture,
            hparams=hparams,
        )
        all_market_encoded_series[market] = market_result_df
        all_market_weights[market] = market_weights

    print("\n--- All Markets Processed ---")
    print(f"Results dictionary keys: {list(all_market_encoded_series.keys())}")

    if save:
        _save_pickle(all_market_encoded_series, encoded_path)
        _save_pickle(all_market_weights, weights_path)

    return all_market_encoded_series, all_market_weights


def run_ae_realgarch_spx(
    market_list,
    all_market_encoded_series,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    result_path = RESULTS_DIR / "ae_rgarch_12121_spx.pkl"

    if use_cache and result_path.exists() and not overwrite:
        return _load_pickle(result_path)

    results = []
    for market in market_list:
        results.append(
            rgarch_lg_dr_fit_pred_process(
                market,
                all_market_encoded_series[market],
                returns_insample[market],
                returns_outsample[market],
            )
        )

    if save:
        _save_pickle(results, result_path)

    return results


def collect_ae_results(results, verbose=True):
    return collect_single_measure_results(results, "AE", verbose=verbose)
