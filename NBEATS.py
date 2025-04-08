# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import os
import copy # 用於 Early Stopping 保存最佳模型狀態

# --- 設定隨機種子以確保結果可重現 (Ensemble中會局部修改) ---
DEFAULT_SEED = 1234
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)
torch.cuda.manual_seed(DEFAULT_SEED)
torch.backends.cudnn.deterministic = True

# --- 決定使用的計算裝置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Loss Functions (sMAPE 修改版) ---
def smape_loss(y_pred, y_true, epsilon=1e-8):
    """
    計算 Symmetric Mean Absolute Percentage Error (SMAPE) Loss。
    實現方式更直接對應 N-BEATS 論文 Page 2 的公式 (|err| / (|y| + |ŷ|))。
    Args:
        y_pred (torch.Tensor): 預測值
        y_true (torch.Tensor): 真實值
        epsilon (float): 避免除以零的小常數
    Returns:
        torch.Tensor: SMAPE 損失值 (標量, 範圍 0-200 theoretically)
    """
    numerator = torch.abs(y_pred - y_true)
    denominator_paper = torch.abs(y_pred) + torch.abs(y_true)
    # Note: Adding epsilon to the denominator before division
    loss_term = numerator / (denominator_paper + epsilon)
    # Average over all elements and multiply by 200 (as per paper formula)
    loss = torch.mean(loss_term) * 200.0
    return loss

def mape_loss(y_pred, y_true, epsilon=1e-8):
    """Mean Absolute Percentage Error (MAPE) Loss"""
    loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0
    return loss

# --- N-BEATS Block Definition (同前) ---
class NBEATSBlock(nn.Module):
    """N-BEATS 的基礎 Block"""
    def __init__(self,
                 input_size: int,
                 theta_dim: int,
                 basis_function: nn.Module,
                 num_hidden_layers: int,
                 hidden_units: int,
                 output_size: int,
                 backcast_length: int):
        super().__init__()
        self.theta_dim = theta_dim
        self.output_size = output_size
        self.backcast_length = backcast_length
        self.basis_function = basis_function

        fc_stack = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            fc_stack.append(nn.Linear(hidden_units, hidden_units))
            fc_stack.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*fc_stack)

        self.theta_b_fc = nn.Linear(hidden_units, theta_dim, bias=False)
        self.theta_f_fc = nn.Linear(hidden_units, theta_dim, bias=False)

    def forward(self, x):
        hidden_output = self.fc_stack(x)
        theta_b = self.theta_b_fc(hidden_output)
        theta_f = self.theta_f_fc(hidden_output)
        backcast, forecast = self.basis_function(theta_b, theta_f)
        return backcast, forecast

# --- N-BEATS Basis Functions (Seasonality 包含常數項) ---
class GenericBasis(nn.Module):
    def __init__(self, theta_dim: int, backcast_length: int, forecast_length: int):
        super().__init__()
        self.backcast_basis = nn.Linear(theta_dim, backcast_length, bias=False)
        self.forecast_basis = nn.Linear(theta_dim, forecast_length, bias=False)

    def forward(self, theta_b, theta_f):
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        return backcast, forecast

class TrendBasis(nn.Module):
    def __init__(self, degree: int, backcast_length: int, forecast_length: int):
        super().__init__()
        self.degree = degree
        self.theta_dim = degree + 1 # Theta 維度等於多項式次數 + 1

        t_forecast = torch.arange(forecast_length, dtype=torch.float32) / forecast_length
        self.T_forecast = torch.stack([t_forecast**i for i in range(degree + 1)], dim=1).to(device)

        t_backcast = torch.arange(backcast_length, dtype=torch.float32) / backcast_length
        self.T_backcast = torch.stack([t_backcast**i for i in range(degree + 1)], dim=1).to(device)

    def forward(self, theta_b, theta_f):
        backcast = torch.matmul(theta_b, self.T_backcast.T)
        forecast = torch.matmul(theta_f, self.T_forecast.T)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, backcast_length: int, forecast_length: int):
        super().__init__()
        # 論文 P5: The seasonality is modelled using Fourier series:
        # y_t = sum_{i=1}^{floor(H/2)} [ a_i * cos(2*pi*i*t) + b_i * sin(2*pi*i*t) ]
        # 這暗示沒有獨立的常數項 a_0 (或被包含在 Trend 中)
        # 但為了更一般性，我們之前的版本加入了常數項。
        # 這裡我們遵循論文 P5 的公式，不加入獨立常數項 a_0，讓 theta_dim = 2 * floor(H/2)
        # 如果需要常數項，可以由 Trend stack 提供，或修改這裡的 basis

        N_f = forecast_length // 2 # Number of Fourier pairs (i=1 to N_f)
        frequencies = torch.arange(1, N_f + 1, dtype=torch.float32) # Frequencies 1, 2, ..., N_f
        self.theta_dim = 2 * N_f # a_i and b_i for i=1 to N_f

        # Create fixed Fourier basis S (without constant term)
        t_forecast = torch.arange(forecast_length, dtype=torch.float32) / forecast_length # Normalize time? Or use absolute time? Let's try 0..H-1
        t_forecast_abs = torch.arange(forecast_length, dtype=torch.float32)
        # Using relative time t (0 to 1) might generalize better? Paper doesn't specify t domain clearly for basis. Let's stick to relative.
        # t_forecast = torch.linspace(0, 1, forecast_length) # Alternative relative time
        S_cos_f = torch.cos(2 * np.pi * frequencies[:, None] * t_forecast[None, :]) # [N_f, forecast_length]
        S_sin_f = torch.sin(2 * np.pi * frequencies[:, None] * t_forecast[None, :]) # [N_f, forecast_length]
        self.S_forecast = torch.cat([S_cos_f, S_sin_f], dim=0).T.to(device) # [forecast_length, 2N_f]

        # For backcast, use negative relative time? Or absolute time? Let's use relative backcast time.
        t_backcast = -torch.arange(backcast_length - 1, -1, -1, dtype=torch.float32) / backcast_length # -1 + 1/L, ..., 0 ? or -(L-1)/L .. 0?
        # Let's try time from -1 to 0 for backcast?
        # t_backcast = torch.linspace(-1, 0, backcast_length) # Seems more standard
        # Need to be careful with basis definition consistency. Let's use absolute indices for now.
        t_backcast_abs = -torch.arange(backcast_length - 1, -1, -1, dtype=torch.float32)

        S_cos_b = torch.cos(2 * np.pi * frequencies[:, None] * t_backcast_abs[None, :]) # [N_f, backcast_length]
        S_sin_b = torch.sin(2 * np.pi * frequencies[:, None] * t_backcast_abs[None, :]) # [N_f, backcast_length]
        self.S_backcast = torch.cat([S_cos_b, S_sin_b], dim=0).T.to(device) # [backcast_length, 2N_f]

    def forward(self, theta_b, theta_f):
        # theta_b/f shape: [batch_size, 2*N_f]
        # Split theta into a_i (for cos) and b_i (for sin) coefficients
        theta_f_cos = theta_f[:, :self.theta_dim // 2] # First N_f are a_i
        theta_f_sin = theta_f[:, self.theta_dim // 2:] # Last N_f are b_i

        theta_b_cos = theta_b[:, :self.theta_dim // 2]
        theta_b_sin = theta_b[:, self.theta_dim // 2:]

        # Access basis vectors correctly
        S_forecast_cos = self.S_forecast[:, :self.theta_dim // 2] # Shape: [forecast_length, N_f]
        S_forecast_sin = self.S_forecast[:, self.theta_dim // 2:] # Shape: [forecast_length, N_f]
        S_backcast_cos = self.S_backcast[:, :self.theta_dim // 2]
        S_backcast_sin = self.S_backcast[:, self.theta_dim // 2:]

        # Calculate forecast: sum a_i * cos(...) + sum b_i * sin(...)
        forecast = torch.matmul(theta_f_cos, S_forecast_cos.T) + torch.matmul(theta_f_sin, S_forecast_sin.T)
        backcast = torch.matmul(theta_b_cos, S_backcast_cos.T) + torch.matmul(theta_b_sin, S_backcast_sin.T)

        return backcast, forecast


# --- N-BEATS Model Definition (加入 Early Stopping Fit) ---
class NBEATS(nn.Module):
    """N-BEATS 模型主類別 (單一模型實例)"""
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 stack_types: list = ['generic', 'generic'],
                 num_blocks_per_stack: int = 3,
                 num_hidden_layers_per_block: int = 4,
                 hidden_units: int = 512,
                 thetas_dims: list = [7, 8], # Adjusted dynamically for interpretable stacks
                 trend_poly_degree: int = 2,
                 share_weights_in_stack: bool = False):
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        self.stacks = nn.ModuleList()
        current_input_size = input_chunk_length

        adjusted_thetas_dims = [] # Store actual theta dims used

        for i, stack_type in enumerate(stack_types):
            original_theta_dim = thetas_dims[i] if i < len(thetas_dims) else None # Handle if thetas_dims is shorter
            current_theta_dim = original_theta_dim # Start with specified dim

            # 選擇 Basis Function 並確定實際 theta_dim
            if stack_type == 'generic':
                if current_theta_dim is None: raise ValueError(f"Theta dim must be provided for generic stack {i}")
                basis_function = GenericBasis(current_theta_dim, current_input_size, output_chunk_length)
            elif stack_type == 'trend':
                basis_function = TrendBasis(trend_poly_degree, current_input_size, output_chunk_length)
                current_theta_dim = basis_function.theta_dim # Override with basis requirement
            elif stack_type == 'seasonality':
                basis_function = SeasonalityBasis(current_input_size, output_chunk_length)
                current_theta_dim = basis_function.theta_dim # Override with basis requirement
            else:
                raise ValueError(f"Unknown stack type: {stack_type}")

            # Check if provided theta_dim matches the requirement for interpretable stacks
            if original_theta_dim is not None and stack_type != 'generic' and original_theta_dim != current_theta_dim:
                 print(f"Warning: Provided theta_dim ({original_theta_dim}) for {stack_type} stack {i} "
                       f"does not match basis requirement ({current_theta_dim}). Using required dimension.")

            adjusted_thetas_dims.append(current_theta_dim)

            # 創建 Blocks
            blocks = nn.ModuleList()
            first_block = NBEATSBlock(
                input_size=current_input_size,
                theta_dim=current_theta_dim,
                basis_function=basis_function,
                num_hidden_layers=num_hidden_layers_per_block,
                hidden_units=hidden_units,
                output_size=output_chunk_length,
                backcast_length=current_input_size
            )
            blocks.append(first_block)

            for _ in range(num_blocks_per_stack - 1):
                if share_weights_in_stack:
                    blocks.append(first_block)
                else:
                    block = NBEATSBlock(
                        input_size=current_input_size,
                        theta_dim=current_theta_dim,
                        basis_function=basis_function,
                        num_hidden_layers=num_hidden_layers_per_block,
                        hidden_units=hidden_units,
                        output_size=output_chunk_length,
                        backcast_length=current_input_size
                    )
                    blocks.append(block)
            self.stacks.append(blocks)

        print(f"Initialized NBEATS with Stack Types: {stack_types}, Actual Theta Dims: {adjusted_thetas_dims}")

    def forward(self, backcast_in):
        # Ensure input has the expected length
        if backcast_in.shape[1] != self.input_chunk_length:
             # This should ideally be handled by data loading, but adding a check/slice here
             # print(f"Warning: Input length {backcast_in.shape[1]} does not match model expected {self.input_chunk_length}. Slicing input.")
             backcast_in = backcast_in[:, -self.input_chunk_length:]

        forecast_out = torch.zeros(backcast_in.size(0), self.output_chunk_length, device=backcast_in.device)
        residual_backcast = backcast_in

        for stack_idx, blocks in enumerate(self.stacks):
            stack_forecast = torch.zeros(backcast_in.size(0), self.output_chunk_length, device=backcast_in.device)
            for block_idx, block in enumerate(blocks):
                block_backcast, block_forecast = block(residual_backcast)
                residual_backcast = residual_backcast - block_backcast
                stack_forecast = stack_forecast + block_forecast
            forecast_out = forecast_out + stack_forecast

        return forecast_out

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader, # Validation loader for early stopping
            epochs: int = 100,
            learning_rate: float = 0.001,
            loss_func = smape_loss,
            optimizer_class = optim.Adam,
            patience: int = 10, # Patience for early stopping
            min_delta: float = 0.001, # Minimum change to qualify as improvement
            verbose: int = 10
           ):
        """訓練 N-BEATS 模型 (包含 Early Stopping)"""
        criterion = loss_func
        optimizer = optimizer_class(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        actual_epochs = 0

        print(f"Starting Training (Input Len: {self.input_chunk_length}, Loss: {loss_func.__name__})...")
        for epoch in range(epochs):
            actual_epochs = epoch + 1
            # --- Training Phase ---
            self.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            start_time = time.time()

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                 # Ensure input matches model's expected length (slice if necessary)
                if batch_x.shape[1] != self.input_chunk_length:
                    batch_x = batch_x[:, -self.input_chunk_length:]

                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                num_train_batches += 1
            avg_epoch_train_loss = epoch_train_loss / max(1, num_train_batches) # Avoid division by zero

            # --- Validation Phase ---
            self.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for batch_x_val, batch_y_val in val_loader:
                    batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
                    # Ensure validation input matches model's expected length
                    if batch_x_val.shape[1] != self.input_chunk_length:
                         batch_x_val = batch_x_val[:, -self.input_chunk_length:]

                    predictions_val = self(batch_x_val)
                    val_loss = criterion(predictions_val, batch_y_val)
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1
            avg_epoch_val_loss = epoch_val_loss / max(1, num_val_batches)
            epoch_time = time.time() - start_time

            if actual_epochs % verbose == 0:
                 print(f'Epoch [{actual_epochs}/{epochs}], Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}, Time: {epoch_time:.2f}s')

            # --- Early Stopping Check ---
            # Check if validation loss improved (considering min_delta)
            if avg_epoch_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_epoch_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.state_dict()) # Save the best state
                if verbose <= 5: print(f"  Epoch {actual_epochs}: New best validation loss: {best_val_loss:.4f}. Saving model.")
            else:
                patience_counter += 1
                if verbose <= 5: print(f"  Epoch {actual_epochs}: Val loss ({avg_epoch_val_loss:.4f}) did not improve enough over best ({best_val_loss:.4f}). Counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {actual_epochs}. Best Val Loss: {best_val_loss:.4f}')
                break

        print(f"Training Finished after {actual_epochs} epochs.")
        # Load the best model state found during training
        if best_model_state is not None:
            print("Loading best model weights based on validation loss.")
            self.load_state_dict(best_model_state)
        else:
            print("Warning: No improvement found based on validation loss or patience=0. Using final model state.")


# --- N-BEATS Ensemble Framework (同前) ---
class NBEATSEnsemble:
    """管理 N-BEATS Ensemble 訓練和預測的框架"""
    def __init__(self,
                 output_chunk_length: int,
                 ensemble_configs: list, # List of dicts, each defining an ensemble member config
                 bagging_iterations: int = 3
                 ):
        self.output_chunk_length = output_chunk_length
        self.ensemble_configs = ensemble_configs
        self.bagging_iterations = bagging_iterations
        self.models = [] # List to store trained model instances

    def fit(self,
            full_train_dataset: Dataset, # The complete training dataset
            val_split_ratio: float = 0.2, # Ratio to split for validation
            epochs_per_model: int = 50,
            learning_rate: float = 0.001,
            batch_size: int = 128,
            patience: int = 10,
            min_delta: float = 0.001,
            verbose_fit: int = 10, # Verbosity for individual model fit
            verbose_ensemble: int = 1 # Print progress per model trained
            ):
        """訓練整個 Ensemble"""

        num_models_to_train = len(self.ensemble_configs) * self.bagging_iterations
        print(f"Starting Ensemble Training: {len(self.ensemble_configs)} unique configs, {self.bagging_iterations} bags each. Total models: {num_models_to_train}")
        self.models = []
        model_counter = 0

        # 準備訓練/驗證數據分割器
        num_train_pool = len(full_train_dataset)
        indices = list(range(num_train_pool))
        split = int(np.floor(val_split_ratio * num_train_pool))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Check if validation set is empty
        if not val_indices:
             print("Warning: Validation set is empty based on val_split_ratio. Early stopping will not function.")
             # Optionally, disable patience or handle this case differently
             patience = epochs_per_model # Effectively disable early stopping if no validation

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Create DataLoaders (Note: Dataset provides max_lookback, model slices internally)
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=train_sampler)
        # Only create val_loader if val_indices is not empty
        val_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=val_sampler) if val_indices else None


        for config_idx, config in enumerate(self.ensemble_configs):
            lookback = config['lookback']
            loss_name = config.get('loss', 'smape')
            stack_types = config['stack_types']
            thetas_dims = config.get('thetas_dims', None) # thetas_dims is now optional for interpretable
            num_blocks = config.get('num_blocks_per_stack', 3)
            num_layers = config.get('num_hidden_layers_per_block', 4)
            hidden_units = config.get('hidden_units', 512)
            trend_degree = config.get('trend_poly_degree', 2)
            share_weights = config.get('share_weights_in_stack', False)

            if loss_name == 'smape': loss_func = smape_loss
            elif loss_name == 'mape': loss_func = mape_loss
            else: loss_func = smape_loss

            for bag_iter in range(self.bagging_iterations):
                model_counter += 1
                print(f"\n--- Training Model {model_counter}/{num_models_to_train} ---")
                print(f"Config: Lookback={lookback}, Loss={loss_name}, Stacks={stack_types}, Bag={bag_iter+1}")

                current_seed = DEFAULT_SEED + config_idx * self.bagging_iterations + bag_iter
                random.seed(current_seed); np.random.seed(current_seed); torch.manual_seed(current_seed); torch.cuda.manual_seed(current_seed)

                model = NBEATS(
                    input_chunk_length=lookback,
                    output_chunk_length=self.output_chunk_length,
                    stack_types=stack_types,
                    num_blocks_per_stack=num_blocks,
                    num_hidden_layers_per_block=num_layers,
                    hidden_units=hidden_units,
                    thetas_dims=thetas_dims if thetas_dims else [], # Pass empty list if None
                    trend_poly_degree=trend_degree,
                    share_weights_in_stack=share_weights
                ).to(device)

                # Pass val_loader to fit; fit handles the case where val_loader is None
                model.fit(
                    train_loader=train_loader,
                    val_loader=val_loader, # Pass the single validation loader
                    epochs=epochs_per_model,
                    learning_rate=learning_rate,
                    loss_func=loss_func,
                    patience=patience if val_loader else epochs_per_model, # Disable patience if no val_loader
                    min_delta=min_delta,
                    verbose=verbose_fit
                )
                self.models.append(model)

        random.seed(DEFAULT_SEED); np.random.seed(DEFAULT_SEED); torch.manual_seed(DEFAULT_SEED); torch.cuda.manual_seed(DEFAULT_SEED)
        print("\nEnsemble Training Complete.")

    def predict(self, data_loader: DataLoader):
        if not self.models: raise RuntimeError("Ensemble has not been trained yet.")

        all_model_predictions = []
        all_actuals = []
        print(f"Starting Ensemble Prediction with {len(self.models)} models...")

        first_batch_processed = False
        for i, model in enumerate(self.models):
            model.eval()
            model_preds = []
            current_model_actuals = [] # Only needed for the first model run

            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    # Input slicing is now handled inside NBEATS.forward if needed
                    input_data = batch_x.to(device)
                    batch_y_cpu = batch_y.cpu().numpy() # Get actuals on CPU

                    predictions = model(input_data) # Model handles slicing internally now
                    model_preds.append(predictions.cpu().numpy())

                    if not first_batch_processed: # Collect actuals only once
                        current_model_actuals.append(batch_y_cpu)

            model_predictions_np = np.concatenate(model_preds, axis=0)
            all_model_predictions.append(model_predictions_np)

            if not first_batch_processed:
                all_actuals = np.concatenate(current_model_actuals, axis=0)
                first_batch_processed = True # Mark that actuals are collected

        stacked_predictions = np.stack(all_model_predictions, axis=0)
        median_predictions = np.median(stacked_predictions, axis=0)
        print("Ensemble Prediction Finished.")
        return median_predictions, all_actuals


# --- 資料準備 (同前) ---
class TimeSeriesDataset(Dataset):
    """簡單的時間序列 Dataset for N-BEATS (提供最大回看長度)"""
    def __init__(self, data, max_input_len, output_len):
        self.data = torch.FloatTensor(data)
        self.max_input_len = max_input_len
        self.output_len = output_len
        self.total_len = len(data)
        self._num_samples = self.total_len - self.max_input_len - self.output_len + 1
        if self._num_samples < 0:
             # Provide more context in the error message
             raise ValueError(f"Not enough data points ({self.total_len}) for the specified max_input_len "
                              f"({self.max_input_len}) and output_len ({self.output_len}). "
                              f"Need at least {self.max_input_len + self.output_len} points.")

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        start_idx = idx
        input_end_idx = start_idx + self.max_input_len
        target_end_idx = input_end_idx + self.output_len
        input_seq = self.data[start_idx : input_end_idx]
        target_seq = self.data[input_end_idx : target_end_idx]
        return input_seq, target_seq

def generate_sine_wave(T=1000, noise_level=0.05):
    t = np.arange(0, T)
    signal = 0.6 * np.sin(2 * np.pi * t / 50) + 0.4 * np.sin(2 * np.pi * t / 25)
    noise = noise_level * np.random.randn(T)
    return signal + noise

# --- 主程式範例 (使用 Ensemble, 同前) ---
if __name__ == "__main__":
    # --- Horizon and Data Parameters ---
    H = 25
    data_len = 2000
    val_ratio = 0.15
    test_start_offset = 0

    # --- Ensemble Configuration ---
    lookback_multipliers = [2, 3] # Reduced for quicker demo
    losses_to_train = ['smape', 'mape']
    bagging_per_config = 1 # Reduced for quicker demo

    ensemble_configs = []
    # 1. Generic Stack Configs
    stack_types_g = ['generic'] * 3
    hidden_units_g = 512
    thetas_dims_g = [hidden_units_g // 8] * len(stack_types_g)
    for lookback_mult in lookback_multipliers:
        for loss_name in losses_to_train:
            ensemble_configs.append({
                'lookback': lookback_mult * H, 'loss': loss_name,
                'stack_types': stack_types_g, 'thetas_dims': thetas_dims_g,
                'hidden_units': hidden_units_g, 'share_weights_in_stack': False,
                'model_type': 'generic'
            })
    # 2. Interpretable Stack Configs
    stack_types_i = ['trend', 'seasonality']
    hidden_units_i = 256
    trend_degree_i = 2
    # Let the model determine theta dims for interpretable
    for lookback_mult in lookback_multipliers:
        for loss_name in losses_to_train:
            ensemble_configs.append({
                'lookback': lookback_mult * H, 'loss': loss_name,
                'stack_types': stack_types_i, 'thetas_dims': None, # Let model determine
                'hidden_units': hidden_units_i, 'trend_poly_degree': trend_degree_i,
                'share_weights_in_stack': True, 'model_type': 'interpretable'
            })

    max_lb = max(cfg['lookback'] for cfg in ensemble_configs)
    print(f"Maximum Lookback Length required: {max_lb}")

    # --- Data Preparation ---
    timeseries_data = generate_sine_wave(data_len, noise_level=0.05)
    train_split_idx = int(data_len * 0.7)
    train_pool_data = timeseries_data[:train_split_idx]
    test_data_points = timeseries_data[train_split_idx - max_lb:] # Ensure history for first test point

    full_train_dataset = TimeSeriesDataset(train_pool_data, max_lb, H)
    test_dataset = TimeSeriesDataset(test_data_points, max_lb, H)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Use larger batch for faster testing metric

    # --- Ensemble Training ---
    nbeats_ensemble = NBEATSEnsemble(
        output_chunk_length=H,
        ensemble_configs=ensemble_configs,
        bagging_iterations=bagging_per_config
    )
    nbeats_ensemble.fit(
        full_train_dataset=full_train_dataset, val_split_ratio=val_ratio,
        epochs_per_model=10, # Increase epochs slightly
        learning_rate=0.001, batch_size=128,
        patience=5, min_delta=0.005,
        verbose_fit=5, verbose_ensemble=1
    )

    # --- Ensemble Prediction & Evaluation ---
    ensemble_predictions, ensemble_actuals = nbeats_ensemble.predict(test_loader)
    test_smape_ensemble = smape_loss(torch.tensor(ensemble_predictions), torch.tensor(ensemble_actuals))
    print(f"\nEnsemble Test SMAPE: {test_smape_ensemble.item():.2f}%") # SMAPE range is 0-200

    # --- Visualization ---
    sample_idx_to_plot = 0
    # Need to get the *original* input corresponding to the first test prediction
    # The test_dataset starts samples such that the first prediction aligns with index train_split_idx in original data
    original_input_start_idx = train_split_idx - max_lb
    original_input_end_idx = train_split_idx
    input_for_plot = timeseries_data[original_input_start_idx:original_input_end_idx]
    target_for_plot = timeseries_data[original_input_end_idx : original_input_end_idx + H]

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(max_lb), input_for_plot, label=f'Input Sequence (len={max_lb})', color='gray')
    x_target = np.arange(max_lb, max_lb + H)
    plt.plot(x_target, target_for_plot, label='Ground Truth', color='blue', marker='.')
    plt.plot(x_target, ensemble_predictions[sample_idx_to_plot, :],
             label=f'Ensemble Prediction (SMAPE: {test_smape_ensemble:.2f})', color='purple', linestyle='--') # Updated label format

    plt.title('N-BEATS Ensemble Time Series Forecasting')
    plt.xlabel('Time Step (relative to start of input)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(np.min(input_for_plot), np.min(target_for_plot), np.min(ensemble_predictions[sample_idx_to_plot, :])) - 0.5,
             max(np.max(input_for_plot), np.max(target_for_plot), np.max(ensemble_predictions[sample_idx_to_plot, :])) + 0.5) # Adjust y-lim
    plt.show()
