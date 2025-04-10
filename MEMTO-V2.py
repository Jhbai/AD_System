import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import math
import tqdm # For progress bars
import matplotlib.pyplot as plt
import os
import random

# --- Set environment variable ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. Positional Encoding (Same as before) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 2. TimeSeries VAE Model (Same as before) ---
class TimeSeriesVAE(nn.Module):
    def __init__(self, win_size, n_series, d_model=128, n_heads=4, num_encoder_layers=2,
                 latent_dim=32, decoder_hidden_dim=128, dropout=0.1):
        super(TimeSeriesVAE, self).__init__()
        self.win_size = win_size
        self.n_series = n_series
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Encoder
        self.input_embedding = nn.Linear(n_series, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=win_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='relu', batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Latent Space
        encoder_output_dim = win_size * d_model
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, decoder_hidden_dim)
        self.decoder_hidden = nn.Linear(decoder_hidden_dim, win_size * n_series)
        self.relu = nn.ReLU()

    def encode(self, x):
        x = x.permute(1, 0, 2)
        embedded = self.input_embedding(x) * math.sqrt(self.d_model)
        pos_encoded = self.pos_encoder(embedded)
        encoder_output = self.transformer_encoder(pos_encoded)
        encoder_output_flat = encoder_output.permute(1, 0, 2).reshape(x.size(1), -1)
        mu = self.fc_mu(encoder_output_flat)
        logvar = self.fc_logvar(encoder_output_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden = self.relu(self.decoder_input(z))
        reconstruction_flat = self.decoder_hidden(hidden)
        recon_x = reconstruction_flat.view(-1, self.win_size, self.n_series)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# --- 3. Loss Function (Same as before, maybe adjust beta later) ---
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Ensure reduction is appropriate - 'mean' averages over all elements
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KLD calculation needs to be averaged over the batch
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = mse_loss + beta * kld_loss
    return total_loss, mse_loss, kld_loss

# --- 4. TimeSeries Dataset for Sliding Windows (Same as before) ---
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, stride=1):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
             raise TypeError("Input data must be a NumPy array or PyTorch tensor.")
        if data.dim() != 2:
             raise ValueError("Input data must have shape (seq_len, n_series).")
        self.data = data.float()
        self.window_size = window_size
        self.stride = stride
        self.num_windows = max(0, (len(data) - window_size) // stride + 1)
        if len(data) < window_size:
             print(f"Warning: Data length ({len(data)}) is less than window size ({window_size}). Dataset will be empty.")
             self.num_windows = 0


    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        # This condition should not be strictly necessary with correct __len__ but acts as a safeguard
        if end_idx > len(self.data):
             start_idx = len(self.data) - self.window_size
             end_idx = len(self.data)
        window = self.data[start_idx:end_idx]
        return window

# --- 5. Model Wrapper (Same as before, default beta adjusted) ---
class model_wrapper:
    def __init__(self, win_size, n_series, d_model=128, n_heads=4, num_encoder_layers=2,
                 latent_dim=32, decoder_hidden_dim=128, dropout=0.1,
                 learning_rate=1e-3, beta=0.1, device='cuda'): # Reduced default beta
        self.win_size = win_size
        self.n_series = n_series
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.beta = beta
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = TimeSeriesVAE(
            win_size=win_size, n_series=n_series, d_model=d_model, n_heads=n_heads,
            num_encoder_layers=num_encoder_layers, latent_dim=latent_dim,
            decoder_hidden_dim=decoder_hidden_dim, dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def fit(self, train_data, epochs=10, batch_size=64, stride=1, val_data=None):
        train_dataset = SlidingWindowDataset(train_data, self.win_size, stride)
        if len(train_dataset) == 0:
             print("Error: Training dataset is empty. Check data length, window size, and stride.")
             return None # Return None or raise error if no training data
        # drop_last=True is important if batch size doesn't divide dataset size, avoids variable batch sizes causing issues
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        val_loader = None
        if val_data is not None:
            val_dataset = SlidingWindowDataset(val_data, self.win_size, stride=1)
            if len(val_dataset) > 0:
                 # No need to drop last for validation usually, but handle potential smaller last batch if needed
                val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
            else:
                 print("Warning: Validation dataset is empty.")


        self.model.train()
        history = {'train_loss': [], 'train_mse': [], 'train_kld': [], 'val_loss': [], 'val_mse': [], 'val_kld': []}

        for epoch in range(epochs):
            epoch_loss, epoch_mse, epoch_kld = 0.0, 0.0, 0.0
            # Check if loader is empty (can happen if drop_last=True and dataset smaller than batch_size)
            if len(train_loader) == 0:
                print(f"Warning: Epoch {epoch+1} - Train loader is empty, skipping training epoch.")
                # Append NaN to history to maintain length consistency if needed by plotting later
                history['train_loss'].append(float('nan'))
                history['train_mse'].append(float('nan'))
                history['train_kld'].append(float('nan'))
                # Handle validation history similarly
                history['val_loss'].append(float('nan'))
                history['val_mse'].append(float('nan'))
                history['val_kld'].append(float('nan'))
                continue # Skip to next epoch


            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, mininterval=0.5)

            for batch_idx, data_batch in enumerate(pbar):
                # This check is technically redundant due to drop_last=True, but harmless
                # if data_batch.shape[0] != batch_size and train_loader.drop_last is False: continue
                data_batch = data_batch.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data_batch)
                loss, mse, kld = loss_function(recon_batch, data_batch, mu, logvar, self.beta)

                # Check for NaN/inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                     # Consider stopping training or reducing learning rate if this happens often
                     continue # Skip this batch update


                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kld += kld.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MSE': f"{mse.item():.4f}", 'KLD': f"{kld.item():.4f}"})

            num_batches = len(train_loader) # Number of batches actually processed
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                avg_epoch_mse = epoch_mse / num_batches
                avg_epoch_kld = epoch_kld / num_batches
                history['train_loss'].append(avg_epoch_loss)
                history['train_mse'].append(avg_epoch_mse)
                history['train_kld'].append(avg_epoch_kld)
                print(f"Epoch {epoch+1} Average Train Loss: {avg_epoch_loss:.4f}, MSE: {avg_epoch_mse:.4f}, KLD: {avg_epoch_kld:.4f}")
            else: # Should not happen if check at epoch start works, but safety first
                 history['train_loss'].append(float('nan'))
                 history['train_mse'].append(float('nan'))
                 history['train_kld'].append(float('nan'))


            if val_loader:
                self.model.eval()
                val_loss, val_mse, val_kld = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for data_batch in val_loader:
                        if data_batch.shape[0] == 0: continue
                        data_batch = data_batch.to(self.device)
                        recon_batch, mu, logvar = self.model(data_batch)
                        loss, mse, kld = loss_function(recon_batch, data_batch, mu, logvar, self.beta)
                        # Check for NaN/Inf in validation too
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                             val_loss += loss.item(); val_mse += mse.item(); val_kld += kld.item()
                        else:
                             print(f"Warning: NaN or Inf validation loss detected at epoch {epoch+1}.")
                             # Decide how to handle this, e.g., count batches or use last valid value


                num_val_batches = len(val_loader)
                if num_val_batches > 0:
                    avg_val_loss = val_loss / num_val_batches
                    avg_val_mse = val_mse / num_val_batches
                    avg_val_kld = val_kld / num_val_batches
                    history['val_loss'].append(avg_val_loss)
                    history['val_mse'].append(avg_val_mse)
                    history['val_kld'].append(avg_val_kld)
                    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, KLD: {avg_val_kld:.4f}")
                else:
                    history['val_loss'].append(float('nan'))
                    history['val_mse'].append(float('nan'))
                    history['val_kld'].append(float('nan'))
                    print(f"Epoch {epoch+1} - No batches processed in validation.")
                self.model.train() # Set back to training mode
            else:
                 # Append NaN if no validation loader
                 history['val_loss'].append(float('nan'))
                 history['val_mse'].append(float('nan'))
                 history['val_kld'].append(float('nan'))


        print("Training finished.")
        return history


    def reconstruct(self, data_seq, stride=1, batch_size=128):
        self.model.eval()
        if isinstance(data_seq, np.ndarray): data_seq = torch.from_numpy(data_seq).float()
        if len(data_seq) < self.win_size: return []
        dataset = SlidingWindowDataset(data_seq, self.win_size, stride)
        if len(dataset) == 0: return []
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_reconstructions = []
        with torch.no_grad():
            for batch in data_loader:
                if batch.shape[0] == 0: continue
                batch = batch.to(self.device)
                recon_batch, _, _ = self.model(batch)
                all_reconstructions.extend([win.cpu() for win in recon_batch])
        return all_reconstructions

    def predict(self, data_seq, stride=1, batch_size=128):
        self.model.eval()
        if isinstance(data_seq, np.ndarray): data_seq = torch.from_numpy(data_seq).float()
        if len(data_seq) < self.win_size: return np.array([])
        dataset = SlidingWindowDataset(data_seq, self.win_size, stride)
        if len(dataset) == 0: return np.array([])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        anomaly_scores = []
        with torch.no_grad():
            for batch in data_loader:
                if batch.shape[0] == 0: continue
                batch = batch.to(self.device)
                recon_batch, _, _ = self.model(batch)
                error = batch - recon_batch
                # Calculate L2 norm per window in the batch
                scores_batch = torch.sqrt(torch.sum(error**2, dim=(1, 2))).cpu().numpy()
                anomaly_scores.extend(scores_batch)
        return np.array(anomaly_scores)

    def rebuild(self, data_seq, stride=1, batch_size=128):
        if isinstance(data_seq, np.ndarray): data_seq = torch.from_numpy(data_seq).float()
        seq_len = data_seq.shape[0]
        if seq_len == 0: return torch.empty_like(data_seq)
        reconstructed_windows = self.reconstruct(data_seq, stride=stride, batch_size=batch_size)
        num_windows = len(reconstructed_windows)
        if num_windows == 0: return torch.zeros_like(data_seq, device='cpu') # Return zeros if no windows
        rebuilt_seq = torch.zeros_like(data_seq, dtype=torch.float32, device='cpu')
        counts = torch.zeros(seq_len, dtype=torch.float32, device='cpu')
        for i in range(num_windows):
            window_recon = reconstructed_windows[i]
            start_index = i * stride
            end_index = start_index + self.win_size
            actual_end_index = min(end_index, seq_len)
            actual_win_size = actual_end_index - start_index
            if actual_win_size <= 0: continue
            # Ensure the reconstructed window has enough data points
            if window_recon.shape[0] < actual_win_size:
                print(f"Warning: Reconstructed window {i} too short ({window_recon.shape[0]} < {actual_win_size}). Skipping.")
                continue
            rebuilt_seq[start_index:actual_end_index, :] += window_recon[:actual_win_size, :]
            counts[start_index:actual_end_index] += 1.0
        # Handle cases where counts might be zero (e.g., if stride > window size)
        counts[counts == 0] = 1.0
        rebuilt_final = rebuilt_seq / counts.unsqueeze(1) # Add dimension for broadcasting
        return rebuilt_final


# --- 6. Stepwise Data Generation (Same as before) ---
def generate_stepwise_data(seq_len, noise_std=0.05, p_event_switch=0.01, min_duration=50):
    n_series = 2
    data = np.zeros((seq_len, n_series))
    noise_a_scale = 80 * noise_std
    noise_b_scale = 70 * noise_std

    current_mode_is_true = random.choice([True, False])
    roles_swapped = random.choice([True, False])
    current_duration = 0

    for t in range(seq_len):
        if not roles_swapped:
            if current_mode_is_true: # Event == True
                val_a = 0.0
                val_b = 70.0 + np.random.normal(0, noise_b_scale)
            else: # Event == False
                val_a = 80.0 + np.random.normal(0, noise_a_scale)
                val_b = 100.0
        else: # Roles are swapped
             if current_mode_is_true:
                val_b = 0.0
                val_a = 70.0 + np.random.normal(0, noise_b_scale)
             else:
                val_b = 80.0 + np.random.normal(0, noise_a_scale)
                val_a = 100.0

        data[t, 0] = val_a
        data[t, 1] = val_b

        current_duration += 1
        # Ensure min_duration is positive before checking
        if min_duration > 0 and current_duration >= min_duration:
            if random.random() < p_event_switch:
                if random.random() < 0.5:
                    current_mode_is_true = not current_mode_is_true
                else:
                    roles_swapped = not roles_swapped
                current_duration = 0

    return torch.from_numpy(data).float()

# --- 7. Anomaly Injection (Corrected Loop) ---
def inject_anomalies(data_seq, win_size, num_anomalies=5, anomaly_duration_range=(20, 50)):
    """Injects specified anomalies into the data sequence."""
    seq_len, n_series = data_seq.shape
    data_anomaly = data_seq.clone()
    anomaly_info = [] # Store type, start, end, series_idx

    anomaly_types = ["shape_mutation", "downward_slope", "spike"]

    # Ensure anomaly duration range has integer values
    min_duration, max_duration = anomaly_duration_range
    if not isinstance(max_duration, int):
         max_duration = int(max_duration)
         print(f"Warning: Converted max anomaly duration to int: {max_duration}")


    # Ensure anomalies don't start too close to the end
    max_possible_duration = max_duration
    max_start_idx = seq_len - max_possible_duration - 1 # Max index where longest anomaly can start

    if max_start_idx <= 0:
         print("Warning: Sequence too short to inject anomalies.")
         return data_anomaly, anomaly_info

    attempts = 0
    max_attempts = num_anomalies * 5 # Limit attempts to avoid infinite loops

    while len(anomaly_info) < num_anomalies and attempts < max_attempts:
        attempts += 1
        anomaly_type = random.choice(anomaly_types)
        # Ensure start index is valid
        if max_start_idx <= win_size // 2:
             print("Warning: max_start_idx too small, adjusting anomaly start range.")
             current_max_start = max(0, seq_len - max_possible_duration - 1) # Recalculate if needed
             if current_max_start <=0: break # Cannot place anomaly
             start_idx = random.randint(0, current_max_start)
        else:
             start_idx = random.randint(win_size // 2, max_start_idx)

        duration = random.randint(min_duration, max_duration) # Use corrected max_duration
        end_idx = min(start_idx + duration, seq_len)
        # Recalculate actual duration if end_idx was clipped
        actual_duration = end_idx - start_idx
        if actual_duration <= 0 : continue # Skip if duration became zero

        series_idx = random.randint(0, n_series - 1)

        # --- Overlap Check ---
        is_overlapping = False
        # Correctly iterate through dictionaries in anomaly_info
        for existing_anomaly in anomaly_info:
            existing_start = existing_anomaly['start']
            existing_end = existing_anomaly['end']
            # Simple check: if the new start falls within an existing range
            if start_idx >= existing_start and start_idx < existing_end:
                is_overlapping = True
                break
            # Optional more complex check: Check for any overlap
            # if max(start_idx, existing_start) < min(end_idx, existing_end):
            #    is_overlapping = True
            #    break
        if is_overlapping:
            # print(f"Skipping overlapping anomaly at {start_idx}") # Optional debug print
            continue # Skip if start falls within another anomaly

        print(f"Injecting {anomaly_type} at [{start_idx}:{end_idx}] in series {series_idx}")

        if anomaly_type == "shape_mutation":
            current_val = data_anomaly[start_idx, series_idx]
            target_val = 0.0 if current_val > 50 else 120.0
            data_anomaly[start_idx:end_idx, series_idx] = target_val

        elif anomaly_type == "downward_slope":
            start_val = data_anomaly[start_idx, series_idx].item() # Get scalar value
            end_val = max(0, start_val * 0.2)
            # Ensure actual_duration is used for linspace
            if actual_duration > 0:
                 slope = torch.linspace(start_val, end_val, steps=actual_duration)
                 data_anomaly[start_idx:end_idx, series_idx] = slope
            else:
                 print(f"Warning: Skipping slope injection due to zero duration at {start_idx}")

        elif anomaly_type == "spike":
             # Spike needs at least 1 step duration
             if actual_duration < 1: continue
             spike_idx = random.randint(start_idx, end_idx - 1)
             # Calculate range safely
             series_data = data_seq[:, series_idx]
             data_range = torch.max(series_data) - torch.min(series_data)
             if data_range.item() == 0: # Handle constant series
                 data_range = torch.abs(torch.mean(series_data)) * 0.5 + 1.0 # Use mean or fallback
             spike_magnitude = data_range * random.choice([1.5, -1.5])
             data_anomaly[spike_idx, series_idx] += spike_magnitude.item()
             # Make spike duration very short
             actual_duration = 2
             end_idx = min(spike_idx + actual_duration, seq_len)
             # Add small decay if possible
             if spike_idx + 1 < seq_len:
                 data_anomaly[spike_idx+1, series_idx] += (spike_magnitude * 0.3).item()


        anomaly_info.append({
            "type": anomaly_type,
            "start": start_idx,
            "end": end_idx, # Use the potentially adjusted end_idx for spike
            "series": series_idx,
            "color": {'shape_mutation': 'red', 'downward_slope': 'purple', 'spike': 'orange'}[anomaly_type]
        })

    if len(anomaly_info) < num_anomalies:
         print(f"Warning: Only able to inject {len(anomaly_info)} out of {num_anomalies} requested anomalies due to overlap or space constraints.")

    return data_anomaly, anomaly_info


# --- 8. Example Usage ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    WIN_SIZE = 50       # Increased window size slightly
    N_SERIES = 2
    SEQ_LEN_TRAIN = 6000
    SEQ_LEN_TEST = 1500
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    LATENT_DIM = 16
    DECODER_HIDDEN = 64
    DROPOUT = 0.1
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    BATCH_SIZE = 64
    BETA = 0.2
    STRIDE_TRAIN = 5
    STRIDE_PREDICT_REBUILD = 1
    MIN_STEP_DURATION = WIN_SIZE * 1.2 # Ensure steps are generally longer than window
    P_EVENT_SWITCH = 0.008
    NOISE_STD = 0.05
    NUM_ANOMALIES_TO_INJECT = 7

    # --- Generate Stepwise Data ---
    print("\n--- Generating Stepwise Data ---")
    # Ensure min_duration is an integer
    min_duration_int = int(MIN_STEP_DURATION)
    train_data = generate_stepwise_data(SEQ_LEN_TRAIN, noise_std=NOISE_STD, p_event_switch=P_EVENT_SWITCH, min_duration=min_duration_int)
    test_data = generate_stepwise_data(SEQ_LEN_TEST, noise_std=NOISE_STD, p_event_switch=P_EVENT_SWITCH, min_duration=min_duration_int)
    time_steps_test = np.arange(SEQ_LEN_TEST)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # --- Instantiate Wrapper and Model ---
    wrapper = model_wrapper(
        win_size=WIN_SIZE, n_series=N_SERIES, d_model=D_MODEL, n_heads=N_HEADS,
        num_encoder_layers=N_LAYERS, latent_dim=LATENT_DIM,
        decoder_hidden_dim=DECODER_HIDDEN, dropout=DROPOUT,
        learning_rate=LEARNING_RATE, beta=BETA, device='cuda'
    )

    # --- Train the Model ---
    print("\n--- Training ---")
    val_split = min(len(test_data)//5, 500) # Use up to 500 points or 20% for validation
    if val_split < WIN_SIZE: # Ensure validation set is at least one window size
        print(f"Warning: Validation split ({val_split}) too small, adjusting.")
        val_split = WIN_SIZE if len(test_data) >= WIN_SIZE else 0

    validation_data = test_data[:val_split] if val_split > 0 else None
    history = wrapper.fit(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, stride=STRIDE_TRAIN, val_data=validation_data)

    # --- Inject Anomalies into Test Data ---
    print("\n--- Injecting Anomalies ---")
    # Correct anomaly duration range passing
    anomaly_duration_range = (WIN_SIZE // 2, int(WIN_SIZE * 1.5))
    test_data_anomaly, anomaly_info = inject_anomalies(test_data, WIN_SIZE, num_anomalies=NUM_ANOMALIES_TO_INJECT, anomaly_duration_range=anomaly_duration_range)

    # --- Rebuild NORMAL Test Data ---
    print("\n--- Rebuilding Normal Test Sequence ---")
    rebuilt_sequence_normal = wrapper.rebuild(test_data, stride=STRIDE_PREDICT_REBUILD, batch_size=BATCH_SIZE*2)

    # --- Predict Anomaly Scores on NORMAL Test Data ---
    print("\n--- Predicting Anomaly Scores on Normal Data ---")
    anomaly_scores_normal = wrapper.predict(test_data, stride=STRIDE_PREDICT_REBUILD, batch_size=BATCH_SIZE*2)
    num_expected_scores = max(0, (SEQ_LEN_TEST - WIN_SIZE) // STRIDE_PREDICT_REBUILD + 1)

    # --- Rebuild ANOMALOUS Test Data ---
    print("\n--- Rebuilding Anomalous Test Sequence ---")
    rebuilt_sequence_anomaly = wrapper.rebuild(test_data_anomaly, stride=STRIDE_PREDICT_REBUILD, batch_size=BATCH_SIZE*2)

    # --- Predict Anomaly Scores on ANOMALOUS Test Data ---
    print("\n--- Predicting Anomaly Scores on Anomalous Data ---")
    anomaly_scores_anomaly = wrapper.predict(test_data_anomaly, stride=STRIDE_PREDICT_REBUILD, batch_size=BATCH_SIZE*2)

    # --- VISUALIZATION ---
    print("\n--- Generating Visualizations ---")

    # Add checks for empty results before plotting
    can_plot_normal = rebuilt_sequence_normal.shape == test_data.shape and len(anomaly_scores_normal) > 0
    can_plot_anomaly = rebuilt_sequence_anomaly.shape == test_data_anomaly.shape and len(anomaly_scores_anomaly) > 0
    can_plot_scores = len(anomaly_scores_normal) == num_expected_scores and len(anomaly_scores_anomaly) == num_expected_scores

    if not (history and can_plot_normal and can_plot_anomaly and can_plot_scores):
        print("Skipping visualization due to missing data, shape mismatch, or training failure.")
    else:
        test_data_np = test_data.cpu().numpy()
        rebuilt_normal_np = rebuilt_sequence_normal.cpu().numpy()
        test_data_anomaly_np = test_data_anomaly.cpu().numpy()
        rebuilt_anomaly_np = rebuilt_sequence_anomaly.cpu().numpy()

        fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)

        # Plot 1: Normal Reconstruction
        ax = axes[0]
        ax.plot(time_steps_test, test_data_np[:, 0], label=f'Original A (Normal)', color='blue', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps_test, rebuilt_normal_np[:, 0], label=f'Reconstructed A (Normal)', color='cyan', linestyle='--', alpha=0.8)
        ax.plot(time_steps_test, test_data_np[:, 1], label=f'Original B (Normal)', color='green', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps_test, rebuilt_normal_np[:, 1], label=f'Reconstructed B (Normal)', color='lime', linestyle='--', alpha=0.8)
        ax.set_ylabel('Value')
        ax.set_title('Normal Data Reconstruction (Series A & B)')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':')

        # Plot 2: Anomalous Reconstruction
        ax = axes[1]
        ax.plot(time_steps_test, test_data_anomaly_np[:, 0], label=f'Original A (Anomaly)', color='blue', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps_test, rebuilt_anomaly_np[:, 0], label=f'Reconstructed A (Anomaly)', color='cyan', linestyle='--', alpha=0.8)
        ax.plot(time_steps_test, test_data_anomaly_np[:, 1], label=f'Original B (Anomaly)', color='green', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps_test, rebuilt_anomaly_np[:, 1], label=f'Reconstructed B (Anomaly)', color='lime', linestyle='--', alpha=0.8)
        plotted_anomaly_labels = set()
        for info in anomaly_info:
             # Ensure indices are within bounds for plotting
             plot_start_idx = info['start']
             plot_end_idx = min(info['end'], len(time_steps_test)-1)
             if plot_start_idx > plot_end_idx: continue # Skip if start > end

             label = f"Anomaly ({info['type']})" if info['type'] not in plotted_anomaly_labels else None
             ax.axvspan(time_steps_test[plot_start_idx], time_steps_test[plot_end_idx],
                        color=info['color'], alpha=0.25, label=label)
             plotted_anomaly_labels.add(info['type'])
        ax.set_ylabel('Value')
        ax.set_title('Anomalous Data Reconstruction (Series A & B)')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':')

        # Plot 3: Anomaly Scores
        ax = axes[2]
        # Ensure score axis aligns correctly with window start times
        score_indices = np.arange(len(anomaly_scores_normal)) * STRIDE_PREDICT_REBUILD
        valid_score_indices = score_indices < len(time_steps_test) # Filter indices beyond test data length
        score_time_axis = time_steps_test[score_indices[valid_score_indices]]
        valid_scores_normal = anomaly_scores_normal[valid_score_indices]
        valid_scores_anomaly = anomaly_scores_anomaly[valid_score_indices]

        ax.plot(score_time_axis, valid_scores_normal, label='Anomaly Scores (Normal Data)', color='black', alpha=0.6, linewidth=1)
        ax.plot(score_time_axis, valid_scores_anomaly, label='Anomaly Scores (Anomalous Data)', color='red', alpha=0.9, linewidth=1.5)
        plotted_anomaly_labels_score = set()
        for info in anomaly_info:
             plot_start_idx = info['start']
             plot_end_idx = min(info['end'], len(time_steps_test)-1)
             if plot_start_idx > plot_end_idx: continue

             label = f"Anomaly Period ({info['type']})" if info['type'] not in plotted_anomaly_labels_score else None
             ax.axvspan(time_steps_test[plot_start_idx], time_steps_test[plot_end_idx],
                        color=info['color'], alpha=0.2, label=label)
             plotted_anomaly_labels_score.add(info['type'])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Anomaly Score (L2 Norm)')
        ax.set_title('Anomaly Scores Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':')
        score_max_anomaly = np.max(valid_scores_anomaly) if len(valid_scores_anomaly) > 0 else 1
        score_max_normal = np.max(valid_scores_normal) if len(valid_scores_normal) > 0 else 1
        # Set ylim more robustly
        plot_upper_limit = max(np.percentile(valid_scores_normal, 99) * 1.5 if len(valid_scores_normal) > 0 else 1,
                               np.percentile(valid_scores_anomaly, 99) * 1.1 if len(valid_scores_anomaly) > 0 else 1)
        ax.set_ylim(0, plot_upper_limit)


        plt.tight_layout()
        plt.show()

        # Optional: Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # Filter out NaN values for plotting if training had issues
        epochs_axis = np.arange(len(history['train_loss']))
        valid_train_idx = ~np.isnan(history['train_loss'])
        valid_val_idx = ~np.isnan(history['val_loss'])

        if np.any(valid_train_idx):
             plt.plot(epochs_axis[valid_train_idx], np.array(history['train_loss'])[valid_train_idx], label='Train Loss')
        if np.any(valid_val_idx):
             plt.plot(epochs_axis[valid_val_idx], np.array(history['val_loss'])[valid_val_idx], label='Validation Loss')

        plt.title('Loss during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if np.any(valid_train_idx) or np.any(valid_val_idx): plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        valid_train_mse_idx = ~np.isnan(history['train_mse'])
        valid_val_mse_idx = ~np.isnan(history['val_mse'])
        valid_train_kld_idx = ~np.isnan(history['train_kld'])
        valid_val_kld_idx = ~np.isnan(history['val_kld'])

        if np.any(valid_train_mse_idx): plt.plot(epochs_axis[valid_train_mse_idx], np.array(history['train_mse'])[valid_train_mse_idx], label='Train MSE')
        if np.any(valid_val_mse_idx): plt.plot(epochs_axis[valid_val_mse_idx], np.array(history['val_mse'])[valid_val_mse_idx], label='Validation MSE')
        if np.any(valid_train_kld_idx): plt.plot(epochs_axis[valid_train_kld_idx], np.array(history['train_kld'])[valid_train_kld_idx], label='Train KLD', linestyle=':')
        if np.any(valid_val_kld_idx): plt.plot(epochs_axis[valid_val_kld_idx], np.array(history['val_kld'])[valid_val_kld_idx], label='Validation KLD', linestyle=':')

        plt.title('MSE & KLD during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        if np.any(valid_train_mse_idx) or np.any(valid_val_mse_idx) or np.any(valid_train_kld_idx) or np.any(valid_val_kld_idx): plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
