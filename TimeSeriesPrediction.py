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

# 1. RNN (LSTM/GRU) Encoder
class EncoderRNN(nn.Module):
    """RNN Encoder (LSTM 或 GRU)"""
    def __init__(self, input_dim, hidden_dim, n_layers, rnn_type='LSTM', dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.upper()

        # 根據 rnn_type 選擇 LSTM 或 GRU
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [batch_size, src_len, input_dim]
        # 對輸入進行 dropout (可選)
        # src = self.dropout(src) # 通常不對原始輸入 dropout

        # RNN 前向傳播
        outputs, hidden = self.rnn(src)
        # outputs shape: [batch_size, src_len, hidden_dim] (所有時間步的隱藏狀態)
        # hidden shape (LSTM): (h_n, c_n)
        #   h_n shape: [n_layers, batch_size, hidden_dim] (最後時間步的隱藏狀態)
        #   c_n shape: [n_layers, batch_size, hidden_dim] (最後時間步的細胞狀態)
        # hidden shape (GRU): h_n
        #   h_n shape: [n_layers, batch_size, hidden_dim] (最後時間步的隱藏狀態)

        return hidden

# 2. RNN (LSTM/GRU) Decoder
class DecoderRNN(nn.Module):
    """RNN Decoder (LSTM 或 GRU) - 預測分布參數 (均值 mu, 對數標準差 log_sigma)"""
    def __init__(self, output_dim, hidden_dim, n_layers, rnn_type='LSTM', dropout=0.1):
        super().__init__()
        self.output_dim = output_dim # 這裡的 output_dim 是指原始序列的維度
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.upper()

        # 根據 rnn_type 選擇 LSTM 或 GRU
        # Decoder 的輸入維度是原始序列的維度 (output_dim)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        # 線性層將隱藏狀態映射到預測的分布參數 (mu 和 log_sigma)
        # 輸出維度是 2 * output_dim (每個原始維度對應一個 mu 和一個 log_sigma)
        self.fc_out = nn.Linear(hidden_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input shape: [batch_size, 1, output_dim] (單個時間步的輸入)
        # hidden shape (LSTM): (h_n, c_n) from encoder or previous step
        # hidden shape (GRU): h_n from encoder or previous step

        # 對輸入進行 dropout
        input = self.dropout(input)

        # RNN 前向傳播 (單個時間步)
        output, hidden = self.rnn(input, hidden)
        # output shape: [batch_size, 1, hidden_dim]
        # hidden shape: updated hidden/cell states

        # 將 RNN 輸出通過線性層得到預測參數
        # output.squeeze(1) shape: [batch_size, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))
        # prediction shape: [batch_size, output_dim * 2]

        # 分割預測的 mu 和 log_sigma
        pred_mu = prediction[:, :self.output_dim]
        pred_log_sigma = prediction[:, self.output_dim:]

        # pred_mu shape: [batch_size, output_dim]
        # pred_log_sigma shape: [batch_size, output_dim]

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
        # teacher_forcing_ratio: 介於 0 到 1 之間，決定使用 teacher forcing 的機率

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_output_dim = self.decoder.output_dim

        # 儲存 decoder 輸出的 tensor (mu 和 log_sigma)
        outputs_mu = torch.zeros(batch_size, trg_len, trg_output_dim).to(self.device)
        outputs_log_sigma = torch.zeros(batch_size, trg_len, trg_output_dim).to(self.device)

        # Encoder 最後的隱藏狀態作為 Decoder 的初始隱藏狀態
        hidden = self.encoder(src)

        # Decoder 的第一個輸入是目標序列的第一個時間步的值
        # 在時間序列預測中，通常使用輸入序列的最後一個值或一個特殊的 <SOS> 標記
        # 這裡我們使用輸入序列的最後一個真實值作為起始（更簡單的做法）
        # 或者，為了與 teacher forcing/non-forcing 邏輯一致，可以從 0 開始或用目標序列的第一个值
        # 我們採用目標序列的第一個值（在 teacher forcing 時），或用零向量（在 non-teacher forcing 時）
        # 這裡為了簡化，我們在循環內部處理第一個輸入
        # 取 src 的最後一個時間步作為 decoder 的第一個輸入
        # decoder_input = src[:,-1:,:] # shape: [batch_size, 1, input_dim]
        # 注意：decoder 的輸入維度是 output_dim，所以可能需要調整
        # 假設 input_dim == output_dim
        # 更通用的方法是使用一個可學習的 <SOS> 或零向量
        decoder_input = torch.zeros(batch_size, 1, trg_output_dim).to(self.device) # Start with zeros

        # 逐個時間步生成預測值
        for t in range(trg_len):
            # Decoder 進行單步預測
            pred_mu_t, pred_log_sigma_t, hidden = self.decoder(decoder_input, hidden)

            # 儲存當前時間步的預測 (mu 和 log_sigma)
            outputs_mu[:, t, :] = pred_mu_t
            outputs_log_sigma[:, t, :] = pred_log_sigma_t

            # 決定是否使用 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 獲取下一個時間步的輸入
            if teacher_force:
                # Teacher forcing: 使用真實的目標值作為下一個輸入
                decoder_input = trg[:, t, :].unsqueeze(1) # shape: [batch_size, 1, output_dim]
            else:
                # Non-teacher forcing: 使用模型自己的預測 (mu) 作為下一個輸入
                # 注意：我們用預測的 mu 而不是抽樣值，這是常見做法
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
        pe = pe.unsqueeze(0).transpose(0, 1) # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # 註冊為 buffer，這樣模型保存時會包含它，但不會被視為模型參數

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 5. Transformer Model
class TransformerModel(nn.Module):
    """基於 nn.Transformer 的 Seq2Seq 模型"""
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # 輸入和輸出的線性嵌入層，將原始維度映射到 d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)

        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # PyTorch 內建的 Transformer 模塊
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # 使用 batch_first=True
        )

        # 最終的線性層，將 Transformer Decoder 的輸出映射到預測的分布參數 (mu, log_sigma)
        self.fc_out = nn.Linear(d_model, output_dim * 2)

    def _generate_square_subsequent_mask(self, sz):
        """生成方陣遮罩，用於防止 Decoder 看到未來的信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        # src shape: [batch_size, src_len, input_dim]
        # trg shape: [batch_size, trg_len, output_dim] (Decoder 的輸入，通常是目標序列向右移一位)

        # 嵌入和位置編碼
        # src_embedded shape: [batch_size, src_len, d_model]
        src_embedded = self.pos_encoder(self.encoder_embedding(src) * math.sqrt(self.d_model))
        # trg_embedded shape: [batch_size, trg_len, d_model]
        trg_embedded = self.pos_encoder(self.decoder_embedding(trg) * math.sqrt(self.d_model))

        # 創建遮罩
        # src_mask: 通常不需要，除非有 padding
        # trg_mask: 必須有，防止 attend to future tokens
        # memory_mask: 通常不需要
        src_len = src.shape[1]
        trg_len = trg.shape[1]
        trg_mask = self._generate_square_subsequent_mask(trg_len).to(device)
        # src_key_padding_mask: 如果有 padding 則需要
        # tgt_key_padding_mask: 如果有 padding 則需要
        # memory_key_padding_mask: 如果有 padding 則需要

        # Transformer 前向傳播
        # output shape: [batch_size, trg_len, d_model]
        output = self.transformer(src_embedded, trg_embedded,
                                  # src_mask=None, # PyTorch >= 1.9
                                  tgt_mask=trg_mask,
                                  # memory_mask=None, # PyTorch >= 1.9
                                  # src_key_padding_mask=None,
                                  # tgt_key_padding_mask=None,
                                  # memory_key_padding_mask=None
                                  )

        # 線性層輸出預測參數
        # prediction shape: [batch_size, trg_len, output_dim * 2]
        prediction = self.fc_out(output)

        # 分割 mu 和 log_sigma
        pred_mu = prediction[..., :self.output_dim]
        pred_log_sigma = prediction[..., self.output_dim:]

        # pred_mu shape: [batch_size, trg_len, output_dim]
        # pred_log_sigma shape: [batch_size, trg_len, output_dim]
        return pred_mu, pred_log_sigma

# --- Gaussian Negative Log Likelihood Loss ---
def gaussian_nll_loss(mu, log_sigma, target, reduction='mean', eps=1e-6):
    """
    計算高斯分佈的負對數似然損失。
    Args:
        mu (torch.Tensor): 預測的均值，shape [batch_size, seq_len, output_dim]
        log_sigma (torch.Tensor): 預測的對數標準差，shape [batch_size, seq_len, output_dim]
        target (torch.Tensor): 真實目標值，shape [batch_size, seq_len, output_dim]
        reduction (str): 'mean' or 'sum' or 'none'
        eps (float): 防止 log(0) 或除以 0 的小常數
    Returns:
        torch.Tensor: 計算出的損失
    """
    sigma = torch.exp(log_sigma) + eps # 計算標準差，加上 eps 防止 sigma 為 0
    # sigma = torch.clamp(sigma, min=eps) # 另一種方法確保 sigma > 0

    # 計算高斯分佈的 log pdf
    log_prob = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (sigma**2)
    )

    # 計算負對數似然
    nll = -log_prob

    # 根據 reduction 參數處理損失
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
        self.data = torch.FloatTensor(data).unsqueeze(-1) # [total_len, 1] -> [total_len, feature_dim=1]
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = len(data)

    def __len__(self):
        # 總樣本數 = 總長度 - 輸入長度 - 輸出長度 + 1
        return self.total_len - self.input_len - self.output_len + 1

    def __getitem__(self, idx):
        # 輸入序列：從 idx 到 idx + input_len
        input_seq = self.data[idx : idx + self.input_len]
        # 目標序列：從 idx + input_len 到 idx + input_len + output_len
        target_seq = self.data[idx + self.input_len : idx + self.input_len + self.output_len]
        # input_seq shape: [input_len, 1]
        # target_seq shape: [output_len, 1]
        return input_seq, target_seq

def generate_sine_wave(T=1000, noise_level=0.05):
    """生成帶噪聲的正弦波數據"""
    t = np.arange(0, T)
    # 創建兩個頻率不同的正弦波疊加
    signal = 0.6 * np.sin(2 * np.pi * t / 50) + 0.4 * np.sin(2 * np.pi * t / 25)
    noise = noise_level * np.random.randn(T)
    return signal + noise

# --- 訓練函數 ---
def train_rnn(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    """訓練 RNN (LSTM/GRU) Seq2Seq 模型一個 epoch"""
    model.train() # 設置模型為訓練模式
    epoch_loss = 0

    for src, trg in dataloader:
        src = src.to(device) # [batch_size, src_len, input_dim]
        trg = trg.to(device) # [batch_size, trg_len, output_dim]

        optimizer.zero_grad() # 清除梯度

        # 前向傳播
        outputs_mu, outputs_log_sigma = model(src, trg, teacher_forcing_ratio)
        # outputs_mu/log_sigma: [batch_size, trg_len, output_dim]
        # trg: [batch_size, trg_len, output_dim]

        # 計算損失 (確保維度匹配)
        # 我們需要比較從第一個預測時間步開始的輸出和目標
        loss = criterion(outputs_mu, outputs_log_sigma, trg)

        loss.backward() # 反向傳播計算梯度
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # 梯度裁剪 (可選)
        optimizer.step() # 更新模型參數

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def train_transformer(model, dataloader, optimizer, criterion, device):
    """訓練 Transformer Seq2Seq 模型一個 epoch"""
    model.train() # 設置模型為訓練模式
    epoch_loss = 0

    for src, trg in dataloader:
        src = src.to(device) # [batch_size, src_len, input_dim]
        trg = trg.to(device) # [batch_size, trg_len, output_dim]

        optimizer.zero_grad() # 清除梯度

        # Transformer Decoder 需要 'shifted right' 的目標序列作為輸入
        # 這意味著第一個輸入是 <SOS> (或零)，後續是真實目標值 t, t+1, ... T-1
        # 目標用於計算損失的是 t+1, t+2, ... T
        # 在時間序列中，可以簡單地使用 trg[:, :-1, :] 作為 decoder 輸入， trg[:, 1:, :] 作為損失目標
        # 但這裡我們預測的是 trg 的所有步，損失也對應所有步
        # 所以我們創建 decoder input，第一個時間步用零，後面用 target
        trg_input = torch.cat((torch.zeros(trg.shape[0], 1, trg.shape[2]).to(device), trg[:, :-1, :]), dim=1)
        # trg_input shape: [batch_size, trg_len, output_dim]

        # 前向傳播
        outputs_mu, outputs_log_sigma = model(src, trg_input)
        # outputs_mu/log_sigma: [batch_size, trg_len, output_dim]

        # 計算損失，與完整的 trg 比較
        loss = criterion(outputs_mu, outputs_log_sigma, trg)

        loss.backward() # 反向傳播計算梯度
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # 梯度裁剪 (可選)
        optimizer.step() # 更新模型參數

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# --- 預測函數 ---
def predict_rnn(model, input_sequence, target_len, device):
    """使用 RNN 模型進行預測 (Non-Teacher Forcing)"""
    model.eval() # 設置模型為評估模式
    input_sequence = input_sequence.to(device) # [batch_size=1, src_len, input_dim]

    with torch.no_grad(): # 不計算梯度
        batch_size = input_sequence.shape[0] # 應該是 1
        input_dim = input_sequence.shape[2] # 這裡需要的是 output_dim
        output_dim = model.decoder.output_dim # 獲取 decoder 的 output_dim

        # Encoder 計算上下文
        hidden = model.encoder(input_sequence)

        # 儲存預測結果的 tensor
        predictions_mu = torch.zeros(batch_size, target_len, output_dim).to(device)

        # 使用輸入序列的最後一個點或零向量作為 Decoder 的第一個輸入
        # decoder_input = input_sequence[:, -1:, :] # 需要維度匹配
        decoder_input = torch.zeros(batch_size, 1, output_dim).to(device)

        # 逐步生成預測
        for t in range(target_len):
            # Decoder 進行單步預測
            pred_mu_t, _, hidden = model.decoder(decoder_input, hidden) # 預測時不需要 log_sigma

            # 儲存當前時間步的預測均值
            predictions_mu[:, t, :] = pred_mu_t

            # 將當前預測的均值作為下一個時間步的輸入
            decoder_input = pred_mu_t.unsqueeze(1) # shape: [batch_size, 1, output_dim]

    return predictions_mu.squeeze(0).cpu().numpy() # 返回 [target_len, output_dim] 的 numpy 陣列

def predict_transformer(model, input_sequence, target_len, device):
    """使用 Transformer 模型進行預測 (Auto-regressive)"""
    model.eval() # 設置模型為評估模式
    input_sequence = input_sequence.to(device) # [batch_size=1, src_len, input_dim]

    with torch.no_grad():
        batch_size = input_sequence.shape[0]
        output_dim = model.output_dim

        # 初始化 Decoder 的輸入序列 (以零向量開始)
        decoder_input = torch.zeros(batch_size, 1, output_dim).to(device) # [1, 1, output_dim]

        # 儲存預測結果
        predictions_mu = torch.zeros(batch_size, target_len, output_dim).to(device)

        # Transformer 的 Encoder 部分只需要計算一次
        src_embedded = model.pos_encoder(model.encoder_embedding(input_sequence) * math.sqrt(model.d_model))
        # memory shape: [batch_size, src_len, d_model]
        memory = model.transformer.encoder(src_embedded) # 假設 batch_first=True

        # 自迴歸地生成預測序列
        for t in range(target_len):
            # 準備 Decoder 輸入的嵌入和位置編碼
            # decoder_input shape: [batch_size, current_len, output_dim]
            current_len = decoder_input.shape[1]
            trg_embedded = model.pos_encoder(model.decoder_embedding(decoder_input) * math.sqrt(model.d_model))

            # 創建目標序列遮罩 (causal mask)
            tgt_mask = model._generate_square_subsequent_mask(current_len).to(device)

            # Transformer Decoder 前向傳播
            # output shape: [batch_size, current_len, d_model]
            output = model.transformer.decoder(trg_embedded, memory, tgt_mask=tgt_mask)

            # 取 Decoder 輸出的最後一個時間步進行預測
            # last_output shape: [batch_size, d_model]
            last_output = output[:, -1, :]

            # 線性層得到預測參數
            prediction = model.fc_out(last_output) # shape: [batch_size, output_dim * 2]
            pred_mu_t = prediction[:, :output_dim]   # shape: [batch_size, output_dim]
            # pred_log_sigma_t = prediction[:, output_dim:] # 預測時通常只需要 mu

            # 儲存當前預測的 mu
            predictions_mu[:, t, :] = pred_mu_t

            # 將當前預測的 mu 加入到 Decoder 的輸入序列中，準備下一次預測
            # next_decoder_input shape: [batch_size, 1, output_dim]
            next_decoder_input = pred_mu_t.unsqueeze(1)
            # decoder_input shape: [batch_size, current_len + 1, output_dim]
            decoder_input = torch.cat([decoder_input, next_decoder_input], dim=1)

    return predictions_mu.squeeze(0).cpu().numpy() # 返回 [target_len, output_dim] 的 numpy 陣列


# --- 主程式 ---
if __name__ == "__main__":
    # --- 參數設定 ---
    INPUT_DIM = 1       # 輸入特徵維度 (單變量時間序列)
    OUTPUT_DIM = 1      # 輸出特徵維度 (單變量時間序列)
    HIDDEN_DIM = 64     # RNN 隱藏層維度
    N_LAYERS = 2        # RNN 層數
    DROPOUT = 0.1       # Dropout 比例
    INPUT_LEN = 50      # 輸入序列長度
    OUTPUT_LEN = 25     # 預測序列長度 (目標序列長度)
    BATCH_SIZE = 64
    N_EPOCHS = 20
    LEARNING_RATE = 0.001
    CLIP = 1.0          # 梯度裁剪值 (RNN 訓練中常用)

    # Transformer 特定參數
    D_MODEL = 64        # Transformer 的 embedding 維度 (同 HIDDEN_DIM)
    NHEAD = 4           # Transformer 的多頭注意力頭數
    NUM_ENCODER_LAYERS = 2 # Transformer Encoder 層數
    NUM_DECODER_LAYERS = 2 # Transformer Decoder 層數
    DIM_FEEDFORWARD = 128 # Transformer Feedforward 層維度

    # 資料生成和準備
    data_len = 2000
    timeseries_data = generate_sine_wave(data_len, noise_level=0.05)
    train_split = int(data_len * 0.7)
    val_split = int(data_len * 0.85)

    train_data = timeseries_data[:train_split]
    val_data = timeseries_data[train_split:val_split]
    test_data = timeseries_data[val_split:]

    train_dataset = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
    val_dataset = TimeSeriesDataset(val_data, INPUT_LEN, OUTPUT_LEN)
    test_dataset = TimeSeriesDataset(test_data, INPUT_LEN, OUTPUT_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 預測時 batch_size=1

    # --- 模型實例化 ---
    # LSTM
    lstm_encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_model = Seq2SeqRNN(lstm_encoder, lstm_decoder, device).to(device)
    lstm_optimizer_tf = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE) # Teacher Forcing 優化器
    lstm_optimizer_ntf = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE) # Non-Teacher Forcing 優化器

    # GRU
    gru_encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_model = Seq2SeqRNN(gru_encoder, gru_decoder, device).to(device)
    gru_optimizer_tf = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    gru_optimizer_ntf = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)

    # Transformer
    transformer_model = TransformerModel(
        INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)

    # 損失函數
    criterion = gaussian_nll_loss

    print("\n--- Training LSTM (Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(lstm_model, train_dataloader, lstm_optimizer_tf, criterion, device, teacher_forcing_ratio=1.0)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training LSTM (Non-Teacher Forcing) ---")
    # 可以選擇重新初始化模型或繼續訓練
    # 這裡我們重新初始化，以比較從頭開始的訓練效果
    lstm_encoder_ntf = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_decoder_ntf = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'LSTM', DROPOUT).to(device)
    lstm_model_ntf = Seq2SeqRNN(lstm_encoder_ntf, lstm_decoder_ntf, device).to(device)
    lstm_optimizer_ntf = optim.Adam(lstm_model_ntf.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(lstm_model_ntf, train_dataloader, lstm_optimizer_ntf, criterion, device, teacher_forcing_ratio=0.0) # 強制 Non-Teacher Forcing
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')


    print("\n--- Training GRU (Teacher Forcing) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(gru_model, train_dataloader, gru_optimizer_tf, criterion, device, teacher_forcing_ratio=1.0)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training GRU (Non-Teacher Forcing) ---")
    gru_encoder_ntf = EncoderRNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_decoder_ntf = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT).to(device)
    gru_model_ntf = Seq2SeqRNN(gru_encoder_ntf, gru_decoder_ntf, device).to(device)
    gru_optimizer_ntf = optim.Adam(gru_model_ntf.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_rnn(gru_model_ntf, train_dataloader, gru_optimizer_ntf, criterion, device, teacher_forcing_ratio=0.0)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Train Loss: {train_loss:.4f}')

    print("\n--- Training Transformer (Implicit Teacher Forcing) ---")
    # 注意：標準的 Transformer 訓練方式隱含了 Teacher Forcing
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
    src_sample, trg_sample = next(test_iterator)

    # 使用不同模型進行預測 (注意：使用對應訓練好的模型)
    # 這裡我們用 Teacher Forcing 訓練的模型進行預測，因為通常效果更好
    # 如果想用 Non-Teacher Forcing 訓練的模型預測，替換模型變數即可 (e.g., lstm_model_ntf)
    lstm_pred = predict_rnn(lstm_model, src_sample, OUTPUT_LEN, device)
    gru_pred = predict_rnn(gru_model, src_sample, OUTPUT_LEN, device)
    transformer_pred = predict_transformer(transformer_model, src_sample, OUTPUT_LEN, device)

    # 準備繪圖數據
    input_vals = src_sample.squeeze(0).numpy()   # [input_len, 1]
    target_vals = trg_sample.squeeze(0).numpy()  # [output_len, 1]

    # x 軸刻度
    x_input = np.arange(INPUT_LEN)
    x_target = np.arange(INPUT_LEN, INPUT_LEN + OUTPUT_LEN)

    # 繪圖比較
    plt.figure(figsize=(15, 5))
    plt.plot(x_input, input_vals, label='Input Sequence', color='gray')
    plt.plot(x_target, target_vals, label='Ground Truth', color='blue', marker='.')
    plt.plot(x_target, lstm_pred, label='LSTM Prediction (TF Trained)', color='red', linestyle='--')
    plt.plot(x_target, gru_pred, label='GRU Prediction (TF Trained)', color='green', linestyle='--')
    plt.plot(x_target, transformer_pred, label='Transformer Prediction', color='purple', linestyle='--')

    # 如果也想畫 Non-Teacher Forcing 訓練的模型結果
    # lstm_pred_ntf = predict_rnn(lstm_model_ntf, src_sample, OUTPUT_LEN, device)
    # plt.plot(x_target, lstm_pred_ntf, label='LSTM Prediction (NTF Trained)', color='orange', linestyle=':')

    plt.title('Time Series Prediction Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
