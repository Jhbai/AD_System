# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

# --- 設定隨機種子以確保結果可重現 ---
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# --- 決定使用的計算裝置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 模型定義 ---

# 1. RNN (RNN/LSTM/GRU) Encoder
class EncoderRNN(nn.Module):
    """RNN Encoder (RNN, LSTM 或 GRU)"""
    def __init__(self, input_dim, hidden_dim, n_layers, rnn_type='LSTM', dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.upper()

        # 根據 rnn_type 選擇 RNN, LSTM 或 GRU
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'RNN': # <--- 新增 RNN 選項
            self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, nonlinearity='tanh') # 可選 tanh 或 relu
        else:
            raise ValueError("rnn_type must be 'RNN', 'LSTM' or 'GRU'")

        self.dropout = nn.Dropout(dropout) # Dropout 通常加在 RNN 層之間，由 rnn 模塊的 dropout 參數處理

    def forward(self, src):
        # src shape: [batch_size, src_len, input_dim]
        # RNN 前向傳播
        outputs, hidden = self.rnn(src)
        # outputs shape: [batch_size, src_len, hidden_dim]
        # hidden shape (LSTM): (h_n, c_n), h_n/c_n shape: [n_layers, batch_size, hidden_dim]
        # hidden shape (GRU/RNN): h_n, h_n shape: [n_layers, batch_size, hidden_dim]

        # 對於 RNN 和 GRU，hidden 就是最後的隱藏狀態 h_n
        # 對於 LSTM，hidden 是一個包含 (h_n, c_n) 的元組
        return hidden

# 2. RNN (RNN/LSTM/GRU) Decoder
class DecoderRNN(nn.Module):
    """RNN Decoder (RNN, LSTM 或 GRU) - 預測分布參數 (均值 mu, 對數標準差 log_sigma)"""
    def __init__(self, output_dim, hidden_dim, n_layers, rnn_type='LSTM', dropout=0.1):
        super().__init__()
        self.output_dim = output_dim # 這裡的 output_dim 是指原始序列的維度
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.upper()

        # 根據 rnn_type 選擇 RNN, LSTM 或 GRU
        # Decoder 的輸入維度是原始序列的維度 (output_dim)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'RNN': # <--- 新增 RNN 選項
             self.rnn = nn.RNN(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError("rnn_type must be 'RNN', 'LSTM' or 'GRU'")

        # 線性層將隱藏狀態映射到預測的分布參數 (mu 和 log_sigma)
        self.fc_out = nn.Linear(hidden_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input shape: [batch_size, 1, output_dim] (單個時間步的輸入)
        # hidden shape: 來自 encoder 或上一步的隱藏/細胞狀態

        # 對輸入進行 dropout (通常加在這裡)
        input = self.dropout(input)

        # RNN 前向傳播 (單個時間步)
        output, hidden = self.rnn(input, hidden)
        # output shape: [batch_size, 1, hidden_dim]

        # 將 RNN 輸出通過線性層得到預測參數
        prediction = self.fc_out(output.squeeze(1)) # prediction shape: [batch_size, output_dim * 2]

        # 分割預測的 mu 和 log_sigma
        pred_mu = prediction[:, :self.output_dim]      # shape: [batch_size, output_dim]
        pred_log_sigma = prediction[:, self.output_dim:] # shape: [batch_size, output_dim]

        return pred_mu, pred_log_sigma, hidden

# 3. Seq2Seq Wrapper for RNNs
class Seq2SeqRNN(nn.Module):
    """將 EncoderRNN 和 DecoderRNN 組合的 Seq2Seq 模型"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [batch_size, src_len, input_dim]
        # trg shape: [batch_size, trg_len, output_dim]
        # teacher_forcing_ratio: 固定為 1.0 進行 Teacher Forcing 訓練

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_output_dim = self.decoder.output_dim

        # 儲存 decoder 輸出的 tensor (mu 和 log_sigma)
        outputs_mu = torch.zeros(batch_size, trg_len, trg_output_dim).to(self.device)
        outputs_log_sigma = torch.zeros(batch_size, trg_len, trg_output_dim).to(self.device)

        # Encoder 最後的隱藏狀態作為 Decoder 的初始隱藏狀態
        hidden = self.encoder(src)

        # Decoder 的第一個輸入: 對於 Teacher Forcing，使用一個零向量或 <SOS> 標記
        decoder_input = torch.zeros(batch_size, 1, trg_output_dim).to(self.device) # Start with zeros

        # 逐個時間步生成預測值
        for t in range(trg_len):
            # Decoder 進行單步預測
            pred_mu_t, pred_log_sigma_t, hidden = self.decoder(decoder_input, hidden)

            # 儲存當前時間步的預測 (mu 和 log_sigma)
            outputs_mu[:, t, :] = pred_mu_t
            outputs_log_sigma[:, t, :] = pred_log_sigma_t

            # 決定是否使用 teacher forcing (根據傳入的 ratio)
            teacher_force = random.random() < teacher_forcing_ratio

            # 獲取下一個時間步的輸入
            if teacher_force:
                # Teacher forcing: 使用真實的目標值作為下一個輸入
                decoder_input = trg[:, t, :].unsqueeze(1) # shape: [batch_size, 1, output_dim]
            else:
                # Non-teacher forcing: 使用模型自己的預測 (mu) 作為下一個輸入
                decoder_input = pred_mu_t.unsqueeze(1) # shape: [batch_size, 1, output_dim]

        # outputs_mu shape: [batch_size, trg_len, output_dim]
        # outputs_log_sigma shape: [batch_size, trg_len, output_dim]
        return outputs_mu, outputs_log_sigma


# 4. Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """Transformer 的位置編碼"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # shape: [max_len, 1, d_model] -> [max_len, d_model] before unsqueeze
        # 修正: PyTorch Transformer 期待 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        # 如果 batch_first=True, 輸入是 [batch_size, seq_len, d_model], PE 應為 [1, max_len, d_model] 或 [max_len, 1, d_model] 並正確廣播
        # 原始實現 pe = pe.unsqueeze(0).transpose(0, 1) # shape: [max_len, 1, d_model]
        # 如果 batch_first=True，輸入 x 是 [batch_size, seq_len, d_model]，需要調整 PE shape
        pe = pe.squeeze(1).unsqueeze(0) # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] (因為 Transformer batch_first=True)
        # self.pe shape: [1, max_len, d_model]
        # 取出對應序列長度的位置編碼並加入 x
        x = x + self.pe[:, :x.size(1), :] # Broadcasting 會自動處理 batch 維度
        return self.dropout(x)

# 5. Transformer Model
class TransformerModel(nn.Module):
    """基於 nn.Transformer 的 Seq2Seq 模型"""
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, max_len=5000): # 增加 max_len
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len) # 傳遞 max_len

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # 使用 batch_first=True，輸入輸出 shape 為 [batch_size, seq_len, feature_dim]
        )

        self.fc_out = nn.Linear(d_model, output_dim * 2)

    def _generate_square_subsequent_mask(self, sz, device): # <--- 增加 device 參數
        """生成方陣遮罩，用於防止 Decoder 看到未來的信息"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1) # <--- 使用 device
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        # src shape: [batch_size, src_len, input_dim]
        # trg shape: [batch_size, trg_len, output_dim] (Decoder 輸入, 已向右移位)

        # --- Encoder ---
        # src_embedded shape: [batch_size, src_len, d_model]
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded)

        # --- Decoder ---
        # trg_embedded shape: [batch_size, trg_len, d_model]
        trg_embedded = self.decoder_embedding(trg) * math.sqrt(self.d_model)
        trg_pos = self.pos_encoder(trg_embedded)

        # --- Masks ---
        src_len = src.shape[1]
        trg_len = trg.shape[1]
        # Decoder target mask (causal mask)
        # 需要傳遞 device 給 mask 生成函數
        trg_mask = self._generate_square_subsequent_mask(trg_len, src.device).to(src.device) # <--- 使用 src.device
        # Padding masks (如果需要處理 padding，則取消註釋並實現)
        # src_padding_mask = None # 範例： torch.zeros(src.shape[0], src.shape[1]).bool().to(src.device)
        # trg_padding_mask = None # 範例： torch.zeros(trg.shape[0], trg.shape[1]).bool().to(src.device)
        # memory_key_padding_mask = src_padding_mask

        # --- Transformer ---
        # output shape: [batch_size, trg_len, d_model]
        output = self.transformer(src_pos, trg_pos,
                                  tgt_mask=trg_mask,
                                  # src_key_padding_mask=src_padding_mask,
                                  # tgt_key_padding_mask=trg_padding_mask,
                                  # memory_key_padding_mask=memory_key_padding_mask
                                 )

        # --- Output Layer ---
        # prediction shape: [batch_size, trg_len, output_dim * 2]
        prediction = self.fc_out(output)
        pred_mu = prediction[..., :self.output_dim]
        pred_log_sigma = prediction[..., self.output_dim:]

        # pred_mu shape: [batch_size, trg_len, output_dim]
        # pred_log_sigma shape: [batch_size, trg_len, output_dim]
        return pred_mu, pred_log_sigma

# --- Gaussian Negative Log Likelihood Loss ---
def gaussian_nll_loss(mu, log_sigma, target, reduction='mean', eps=1e-6):
    """計算高斯分佈的負對數似然損失"""
    sigma = torch.exp(log_sigma) + eps
    log_prob = -0.5 * (torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (sigma**2))
    nll = -log_prob
    if reduction == 'mean':
        return torch.mean(nll)
    elif reduction == 'sum':
        return torch.sum(nll)
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

# --- 資料準備 ---
class TimeSeriesDataset(Dataset):
    """簡單的時間序列 Dataset"""
    def __init__(self, data, input_len, output_len):
        # 確保 data 是 float tensor
        self.data = torch.FloatTensor(data).unsqueeze(-1) # [total_len, feature_dim=1]
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = len(data)
        if self.total_len < input_len + output_len:
             raise ValueError("Data length is too short for the given input and output lengths.")

    def __len__(self):
        return self.total_len - self.input_len - self.output_len + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.input_len]
        target_seq = self.data[idx + self.input_len : idx + self.input_len + self.output_len]
        return input_seq, target_seq

def generate_sine_wave(T=1000, noise_level=0.05):
    """生成帶噪聲的正弦波數據"""
    t = np.arange(0, T)
    signal = 0.6 * np.sin(2 * np.pi * t / 50) + 0.4 * np.sin(2 * np.pi * t / 25)
    noise = noise_level * np.random.randn(T)
    return signal + noise

# --- 訓練函數 ---
def train_rnn(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    """訓練 RNN (RNN/LSTM/GRU) Seq2Seq 模型一個 epoch"""
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = src.to(device) # [batch_size, src_len, input_dim]
        trg = trg.to(device) # [batch_size, trg_len, output_dim]

        optimizer.zero_grad()

        # 強制使用 Teacher Forcing (teacher_forcing_ratio=1.0)
        outputs_mu, outputs_log_sigma = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        loss = criterion(outputs_mu, outputs_log_sigma, trg)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP) # 如果需要梯度裁剪
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def train_transformer(model, dataloader, optimizer, criterion, device):
    """訓練 Transformer Seq2Seq 模型一個 epoch (隱含 Teacher Forcing)"""
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = src.to(device) # [batch_size, src_len, input_dim]
        trg = trg.to(device) # [batch_size, trg_len, output_dim]

        optimizer.zero_grad()

        # 準備 Decoder 輸入 (shifted right target sequence)
        # 第一個 token 通常是 <SOS> 或 0，最後一個 token 被忽略
        # 創建 decoder input: [batch_size, trg_len, output_dim]
        # 第一個時間步用零向量
        trg_input = torch.cat((torch.zeros(trg.shape[0], 1, trg.shape[2]).to(device), trg[:, :-1, :]), dim=1)

        # 前向傳播
        outputs_mu, outputs_log_sigma = model(src, trg_input)

        # 計算損失 (與原始目標 trg 比較)
        loss = criterion(outputs_mu, outputs_log_sigma, trg)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP) # 如果需要梯度裁剪
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# --- 預測函數 ---
# 注意：預測時總是使用 Non-Teacher Forcing (自迴歸)
def predict_rnn(model, input_sequence, target_len, device):
    """使用 RNN 模型進行預測 (Non-Teacher Forcing / Auto-regressive)"""
    model.eval()
    input_sequence = input_sequence.to(device) # [batch_size=1, src_len, input_dim]

    with torch.no_grad():
        batch_size = input_sequence.shape[0]
        output_dim = model.decoder.output_dim

        hidden = model.encoder(input_sequence)
        predictions_mu = torch.zeros(batch_size, target_len, output_dim).to(device)
        # 使用零向量作為 Decoder 的第一個輸入
        decoder_input = torch.zeros(batch_size, 1, output_dim).to(device)

        for t in range(target_len):
            pred_mu_t, _, hidden = model.decoder(decoder_input, hidden) # 只取 mu
            predictions_mu[:, t, :] = pred_mu_t
            decoder_input = pred_mu_t.unsqueeze(1) # 使用預測的 mu 作為下一步輸入

    return predictions_mu.squeeze(0).cpu().numpy() # [target_len, output_dim]

def predict_transformer(model, input_sequence, target_len, device):
    """使用 Transformer 模型進行預測 (Auto-regressive)"""
    model.eval()
    input_sequence = input_sequence.to(device) # [batch_size=1, src_len, input_dim]

    with torch.no_grad():
        batch_size = input_sequence.shape[0]
        output_dim = model.output_dim
        d_model = model.d_model

        # --- Encoder (只計算一次) ---
        src_embedded = model.encoder_embedding(input_sequence) * math.sqrt(d_model)
        src_pos = model.pos_encoder(src_embedded)
        memory = model.transformer.encoder(src_pos) # memory shape: [batch_size, src_len, d_model]

        # --- Decoder (自迴歸生成) ---
        # 初始化 Decoder 輸入序列 (以零向量開始，模擬 <SOS>)
        decoder_input = torch.zeros(batch_size, 1, output_dim).to(device) # [1, 1, output_dim]
        predictions_mu = torch.zeros(batch_size, target_len, output_dim).to(device)

        for t in range(target_len):
            current_len = decoder_input.shape[1]
            # 嵌入和位置編碼
            trg_embedded = model.decoder_embedding(decoder_input) * math.sqrt(d_model)
            trg_pos = model.pos_encoder(trg_embedded) # shape: [batch_size, current_len, d_model]

            # 創建 causal mask
            tgt_mask = model._generate_square_subsequent_mask(current_len, device).to(device)

            # Transformer Decoder 前向傳播
            output = model.transformer.decoder(trg_pos, memory, tgt_mask=tgt_mask)
            # output shape: [batch_size, current_len, d_model]

            # 取 Decoder 輸出的最後一個時間步進行預測
            last_output = output[:, -1, :] # shape: [batch_size, d_model]

            # 線性層得到預測參數
            prediction = model.fc_out(last_output) # shape: [batch_size, output_dim * 2]
            pred_mu_t = prediction[:, :output_dim]   # shape: [batch_size, output_dim]

            # 儲存當前預測的 mu
            predictions_mu[:, t, :] = pred_mu_t

            # 將當前預測的 mu 加入到 Decoder 的輸入序列中，準備下一次預測
            next_decoder_input = pred_mu_t.unsqueeze(1) # shape: [batch_size, 1, output_dim]
            decoder_input = torch.cat([decoder_input, next_decoder_input], dim=1)

    return predictions_mu.squeeze(0).cpu().numpy() # [target_len, output_dim]


# --- 主程式 ---
if __name__ == "__main__":
    # --- 參數設定 ---
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    HIDDEN_DIM = 64
    N_LAYERS = 2
    DROPOUT = 0.1
    INPUT_LEN = 50
    OUTPUT_LEN = 25
    BATCH_SIZE = 64
    N_EPOCHS = 20 # 可以增加 Epoch 數量以獲得更好效果
    LEARNING_RATE = 0.001
    # CLIP = 1.0 # 如果需要梯度裁剪，取消註釋並在訓練函數中使用

    # Transformer 特定參數
    D_MODEL = 64
    NHEAD = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 128
    MAX_LEN = INPUT_LEN + OUTPUT_LEN + 10 # Positional Encoding 的最大長度

    # 資料生成和準備
    data_len = 2000
    timeseries_data = generate_sine_wave(data_len, noise_level=0.05)
    train_split = int(data_len * 0.7)
    val_split = int(data_len * 0.85)

    train_data = timeseries_data[:train_split]
    val_data = timeseries_data[train_split:val_split] # 驗證集這裡未使用，但保留以便未來擴充
    test_data = timeseries_data[val_split:]

    train_dataset = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
    test_dataset = TimeSeriesDataset(test_data, INPUT_LEN, OUTPUT_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 預測時 batch_size=1

    # --- 模型實例化 ---
    print("\n--- Instantiating Models ---")
    # (1) RNN
    rnn_encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'RNN', DROPOUT).to(device)
    rnn_decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'RNN', DROPOUT).to(device)
    rnn_model = Seq2SeqRNN(rnn_encoder, rnn_decoder, device).to(device)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)

    # (2) LSTM
    lstm_encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_model = Seq2SeqRNN(lstm_encoder, lstm_decoder, device).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    # (3) GRU
    gru_encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_model = Seq2SeqRNN(gru_encoder, gru_decoder, device).to(device)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)

    # (4) Transformer
    transformer_model = TransformerModel(
        INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, MAX_LEN
    ).to(device)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)

    # 損失函數
    criterion = gaussian_nll_loss

    # --- 模型訓練 (全部使用 Teacher Forcing) ---
    teacher_forcing_ratio_train = 1.0 # 固定為 1.0

    print("\n--- Training RNN (Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(rnn_model, train_dataloader, rnn_optimizer, criterion, device, teacher_forcing_ratio_train)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training LSTM (Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(lstm_model, train_dataloader, lstm_optimizer, criterion, device, teacher_forcing_ratio_train)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training GRU (Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(gru_model, train_dataloader, gru_optimizer, criterion, device, teacher_forcing_ratio_train)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training Transformer (Implicit Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_transformer(transformer_model, train_dataloader, transformer_optimizer, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')


    # --- 預測與視覺化 ---
    print("\n--- Generating Predictions ---")
    # 從測試集中取一個樣本進行預測
    test_iterator = iter(test_dataloader)
    try:
        src_sample, trg_sample = next(test_iterator)
    except StopIteration:
        print("Test dataloader is empty.")
        exit()


    # 使用四個訓練好的模型進行預測 (預測時皆為 auto-regressive)
    rnn_pred = predict_rnn(rnn_model, src_sample, OUTPUT_LEN, device)
    lstm_pred = predict_rnn(lstm_model, src_sample, OUTPUT_LEN, device)
    gru_pred = predict_rnn(gru_model, src_sample, OUTPUT_LEN, device)
    transformer_pred = predict_transformer(transformer_model, src_sample, OUTPUT_LEN, device)

    # 準備繪圖數據
    input_vals = src_sample.squeeze(0).cpu().numpy()   # [input_len, 1]
    target_vals = trg_sample.squeeze(0).cpu().numpy()  # [output_len, 1]

    # x 軸刻度
    x_input = np.arange(INPUT_LEN)
    x_target = np.arange(INPUT_LEN, INPUT_LEN + OUTPUT_LEN)

    # 繪圖比較
    plt.figure(figsize=(15, 6)) # 稍微調整圖像大小
    plt.plot(x_input, input_vals[:, 0], label='Input Sequence', color='gray', linewidth=1.5) # 取第一維
    plt.plot(x_target, target_vals[:, 0], label='Ground Truth', color='blue', marker='.', markersize=8, linewidth=2) # 取第一維
    plt.plot(x_target, rnn_pred[:, 0], label='RNN Prediction', color='orange', linestyle='--', linewidth=1.5) # 取第一維
    plt.plot(x_target, lstm_pred[:, 0], label='LSTM Prediction', color='red', linestyle='--', linewidth=1.5) # 取第一維
    plt.plot(x_target, gru_pred[:, 0], label='GRU Prediction', color='green', linestyle='--', linewidth=1.5) # 取第一維
    plt.plot(x_target, transformer_pred[:, 0], label='Transformer Prediction', color='purple', linestyle='--', linewidth=1.5) # 取第一維

    plt.title('Time Series Prediction Comparison (Trained with Teacher Forcing)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='upper right') # 調整圖例位置
    plt.grid(True, linestyle=':', alpha=0.7) # 調整網格樣式
    plt.tight_layout() # 自動調整邊距
    plt.show()
