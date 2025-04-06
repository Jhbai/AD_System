import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.cluster import KMeans

# ---------------------------
# (1) 模擬多變量時間序列資料 (shape: (#Time, #Series))
np.random.seed(42)
time_steps = 1000
t = np.arange(time_steps)
trend = 0.005 * t

series1 = 10 * np.sin(0.02 * t) + trend + np.random.normal(scale=0.5, size=time_steps)
series2 = 5  * np.cos(0.015 * t + 0.5) + trend + np.random.normal(scale=0.3, size=time_steps)
series3 = 8  * np.sin(0.025 * t + 1.0) + trend + np.random.normal(scale=0.7, size=time_steps)
data = np.stack([series1, series2, series3], axis=1)  # shape: (1000, 3)

# ---------------------------
# (2) 將時間序列資料轉換為 (#Batch, #WindowSize, #Series)
def sliding_window(data, window_size, stride=1):
    num_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.array([data[i*stride : i*stride+window_size] for i in range(num_windows)])
    return windows

window_size = 50
windows = sliding_window(data, window_size, stride=1)  # (num_windows, 50, 3)
print("滑動視窗後資料形狀:", windows.shape)

# ---------------------------
# (3) MEMTO-VAE 模型實作（包含 Gated Memory Update、Query Update、Entropy Loss、以及 encode 用於 LSD 計算）
class MEMTOVAE(nn.Module):
    def __init__(self, input_dim=3, window_size=50, model_dim=16, latent_dim=16, 
                 memory_items=10, nhead=4, num_encoder_layers=2, tau=0.1):
        """
        參數說明：
          - input_dim: 原始序列維度 (3)
          - window_size: 視窗長度 (50)
          - model_dim: Transformer 的 d_model (必須可被 nhead 整除，此處設16)
          - latent_dim: VAE 隱空間維度 (16)
          - memory_items: 記憶項數量
          - nhead, num_encoder_layers: Transformer 參數
          - tau: 溫度超參數
        """
        super(MEMTOVAE, self).__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.memory_items = memory_items
        self.tau = tau
        
        # (a) 輸入嵌入：將 (B, L, input_dim) -> (B, L, model_dim)
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # (b) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # (c) Query Update Stage 與 VAE latent mapping：
        # Query 更新後的輸出 shape: (B, L, model_dim*2)
        # 展平後映射到 latent space 得到 mu 與 logvar
        self.fc_mu = nn.Linear(window_size * (model_dim + latent_dim), latent_dim)
        self.fc_logvar = nn.Linear(window_size * (model_dim + latent_dim), latent_dim)
        
        # (d) 記憶模組 (Gated Memory Module)
        self.memory = nn.Parameter(torch.randn(memory_items, latent_dim))
        self.U_psi = nn.Linear(latent_dim, latent_dim)
        self.W_psi = nn.Linear(latent_dim, latent_dim)
        
        # (e) 弱解碼器：從 latent space 重構輸入
        self.decoder_fc1 = nn.Linear(latent_dim, window_size * input_dim)
        self.decoder_fc2 = nn.Linear(window_size * input_dim, window_size * input_dim)
        
        self.relu = nn.ReLU()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        取代 forward 前半部，用於提取嵌入與 Transformer 編碼器的輸出 Q。
        輸出：Q shape (B, L, model_dim)
        """
        x_emb = self.input_embedding(x)  # (B, L, model_dim)
        x_t = x_emb.transpose(0, 1)        # (L, B, model_dim)
        encoded = self.transformer_encoder(x_t)  # (L, B, model_dim)
        Q = encoded.transpose(0, 1)        # (B, L, model_dim)
        return Q

    def memory_update_queries(self, Q):
        """
        Gated Memory Update:
        Q: (B, L, model_dim)
        依據公式：
          v_{i} = softmax( < m_i, Q_flat> / tau )  (over all queries)
          fuse_i = sum_j v_{i,j} * Q_flat[j]
          psi_i = sigmoid( U_psi(m_i) + W_psi(fuse_i) )
          更新： m_i = (1 - psi_i)*m_i + psi_i * fuse_i
        """
        B, L, D = Q.size()
        Q_flat = Q.reshape(B * L, D)  # (B*L, model_dim)
        M = self.memory_items
        updated_memory = []
        for i in range(M):
            m_i = self.memory[i]  # (latent_dim,)
            scores = torch.matmul(Q_flat, m_i) / self.tau  # (B*L,)
            weights = torch.softmax(scores, dim=0)         # (B*L,)
            fuse = torch.sum(weights.unsqueeze(1) * Q_flat, dim=0)  # (model_dim,)
            psi = torch.sigmoid(self.U_psi(m_i) + self.W_psi(fuse))  # (latent_dim,)
            m_updated = (1 - psi) * m_i + psi * fuse
            updated_memory.append(m_updated)
        updated_memory = torch.stack(updated_memory, dim=0)  # (M, latent_dim)
        self.memory.data = updated_memory

    def query_update(self, Q):
        """
        Query Update Stage:
        Q: (B, L, model_dim)
        計算注意力：w = softmax( <q, m> / tau ) (over memory items)
        檢索記憶向量： q_tilde = w @ m
        更新查詢： q_hat = concat(q, q_tilde)
        返回： q_hat: (B, L, model_dim + latent_dim) 與注意力權重 W
        """
        B, L, D = Q.size()
        m = self.memory  # (M, latent_dim)
        scores = torch.matmul(Q, m.transpose(0, 1)) / self.tau  # (B, L, M)
        W = torch.softmax(scores, dim=-1)  # (B, L, M)
        q_tilde = torch.matmul(W, m)        # (B, L, latent_dim)
        q_hat = torch.cat([Q, q_tilde], dim=-1)  # (B, L, model_dim + latent_dim)
        return q_hat, W

    def forward(self, x):
        """
        x: (B, window_size, input_dim)
        """
        B = x.size(0)
        # (a) 利用嵌入層與 Transformer Encoder 得到查詢向量 Q
        Q = self.encode(x)  # (B, window_size, model_dim)
        
        # (b) 若訓練階段，利用 Q 更新記憶模組
        if self.training:
            self.memory_update_queries(Q)
        
        # (c) Query Update Stage：更新查詢並獲得注意力權重（用於 Entropy Loss）
        q_hat, attn_weights = self.query_update(Q)  # (B, window_size, model_dim + latent_dim)
        
        # (d) 將更新後的查詢展平，映射至 latent space 得到 mu 與 logvar
        q_hat_flat = q_hat.reshape(B, -1)  # (B, window_size*(model_dim+latent_dim))
        mu = self.fc_mu(q_hat_flat)         # (B, latent_dim)
        logvar = self.fc_logvar(q_hat_flat)   # (B, latent_dim)
        z = self.reparameterize(mu, logvar)   # (B, latent_dim)
        
        # (e) 弱解碼器：從 z 重構輸入
        dec = self.relu(self.decoder_fc1(z))
        dec = self.decoder_fc2(dec)
        x_hat = dec.reshape(B, self.window_size, self.input_dim)
        
        return x_hat, mu, logvar, attn_weights

    def entropy_loss(self, attn_weights):
        """
        Entropy Loss:
        L_entr = mean( -w * log(w) ) over (B, window_size, memory_items)
        """
        eps = 1e-8
        entr = -attn_weights * torch.log(attn_weights + eps)
        return entr.mean()

    def compute_lsd(self, Q):
        """
        計算 Latent Space Deviation (LSD)
        Q: (B, window_size, model_dim)
        對每個查詢向量 q，計算與記憶中最近的向量的歐氏距離
        返回： LSD: (B, window_size)
        """
        # memory: (M, latent_dim), Q assumed shape (B, L, model_dim) and model_dim == latent_dim
        B, L, D = Q.size()
        m = self.memory  # (M, D)
        # 計算 Q 與 m 之間的距離：使用 torch.cdist
        distances = torch.cdist(Q.reshape(B * L, D), m.unsqueeze(0).expand(B * L, -1, -1))
        # distances: (B*L, M)；取最小值
        lsd, _ = torch.min(distances, dim=1)
        lsd = lsd.reshape(B, L)
        return lsd

# ---------------------------
# (4) 包裝模型：包含 .fit, .predict, .reconstruct 與 compute_anomaly_score
class MEMTOVAEModel:
    def __init__(self, device='cpu', patience=10, **kwargs):
        self.device = device
        self.model = MEMTOVAE(**kwargs).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.criterion = nn.MSELoss(reduction='mean')
        self.lambda_kl = 0.001     # KL divergence 權重
        self.lambda_ent = 0.01     # Entropy Loss 權重
        self.patience = patience   # Early Stopping patience

    def loss_function(self, x, x_hat, mu, logvar, attn_weights):
        rec_loss = self.criterion(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        ent_loss = self.model.entropy_loss(attn_weights)
        return rec_loss + self.lambda_kl * kl_loss + self.lambda_ent * ent_loss

    def initialize_memory(self, train_data, percentage=0.1):
        """
        Two-Phase 訓練：利用隨機抽取的 10% 訓練資料，通過 encoder 得到查詢 Q，
        然後展平並用 KMeans 聚類初始化記憶模組。
        """
        self.model.eval()
        num_samples = int(len(train_data) * percentage)
        indices = np.random.choice(len(train_data), num_samples, replace=False)
        sample_data = train_data[indices]
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            Q = self.model.encode(sample_tensor)  # (N, window_size, model_dim)
            Q_flat = Q.reshape(Q.size(0), -1).cpu().numpy()  # (N, window_size*model_dim)
        kmeans = KMeans(n_clusters=self.model.memory_items, random_state=42)
        kmeans.fit(Q_flat)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
        # 初始化記憶模組 (取前 latent_dim 維)
        self.model.memory.data = centroids[:, :self.model.latent_dim]
        print("記憶模組初始化完成。")

    def fit(self, train_loader, valid_loader=None, epochs=50):
        best_loss = float('inf')
        patience_counter = 0
        
        # 收集所有訓練資料用於記憶初始化
        all_train = []
        for (batch_x,) in train_loader:
            all_train.append(batch_x.numpy())
        all_train = np.concatenate(all_train, axis=0)
        self.initialize_memory(all_train, percentage=0.1)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                x_hat, mu, logvar, attn_weights = self.model(batch_x)
                loss = self.loss_function(batch_x, x_hat, mu, logvar, attn_weights)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)
            
            if valid_loader is not None:
                self.model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for (batch_x,) in valid_loader:
                        batch_x = batch_x.to(self.device)
                        x_hat, mu, logvar, attn_weights = self.model(batch_x)
                        loss = self.loss_function(batch_x, x_hat, mu, logvar, attn_weights)
                        valid_loss += loss.item() * batch_x.size(0)
                valid_loss /= len(valid_loader.dataset)
                print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")
                curr_loss = valid_loss
            else:
                print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.6f}")
                curr_loss = train_loss

            # Early Stopping
            if curr_loss < best_loss:
                best_loss = curr_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        self.model.load_state_dict(best_model_state)
    
    def reconstruct(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            x_hat, _, _, _ = self.model(x)
        return x_hat.cpu().numpy()
    
    def compute_anomaly_score(self, x):
        """
        計算雙維度偏差異常得分：
         - LSD：利用 encoder 得到查詢 Q，再計算每個時間點與記憶中最近記憶的距離。
         - ISD：計算原始輸入與重構輸入之間的 MSE，逐時間點計算。
         - 最終異常得分：對每筆資料，對時間軸進行 softmax(LSD) 之後與 ISD 逐元素相乘。
        返回 anomaly_score: (B, window_size)
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            x_hat, _, _, _ = self.model(x)
            # ISD: (B, window_size) 每個時間點的重構誤差
            isd = torch.mean((x - x_hat) ** 2, dim=-1)
            # 取得 Q 來計算 LSD
            Q = self.model.encode(x)  # (B, window_size, model_dim)
            lsd = self.model.compute_lsd(Q)  # (B, window_size)
            # 對每筆資料沿時間軸做 softmax (normalized LSD)
            lsd_norm = torch.softmax(lsd, dim=1)
            # 異常得分 = lsd_norm ∘ isd (element-wise multiplication)
            anomaly_score = lsd_norm * isd
        return anomaly_score.cpu().numpy()

    def predict(self, x, threshold):
        """
        使用雙維度偏差異常得分進行異常判定：
         - threshold: 門檻值，若平均異常得分超過 threshold 則判為異常
        返回：preds, anomaly_scores (兩者 shape: (B, window_size))
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            x_hat, _, _, _ = self.model(x)
            isd = torch.mean((x - x_hat) ** 2, dim=-1)  # (B, window_size)
            Q = self.model.encode(x)  # (B, window_size, model_dim)
            lsd = self.model.compute_lsd(Q)  # (B, window_size)
            lsd_norm = torch.softmax(lsd, dim=1)
            anomaly_score = lsd_norm * isd
            # 以平均異常得分判斷
            score_avg = anomaly_score.mean(dim=1)
            preds = (score_avg > threshold).float()
        return preds.cpu().numpy(), anomaly_score.cpu().numpy()

# ---------------------------
# (5) 建立資料集與 DataLoader (切分訓練集與驗證集)
batch_size = 32
all_windows = torch.tensor(windows, dtype=torch.float32)
dataset = TensorDataset(all_windows)
dataset_size = len(dataset)
valid_size = int(0.2 * dataset_size)
train_size = dataset_size - valid_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# (6) 訓練模型與繪圖
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper = MEMTOVAEModel(
    device=device,
    input_dim=3,
    window_size=window_size,
    model_dim=16,      # Transformer d_model
    latent_dim=16,
    memory_items=10,
    nhead=4,
    num_encoder_layers=2,
    tau=0.1,
    patience=10
)
model_wrapper.fit(train_loader, valid_loader, epochs=50)

# 使用訓練好的模型重構資料
reconstructed_windows = model_wrapper.reconstruct(all_windows)

# 將重構結果貼回完整時間序列 (示範用：取每個視窗中間值)
reconstructed_full = np.zeros_like(data)
half_win = window_size // 2
for i in range(reconstructed_windows.shape[0]):
    idx = i + half_win
    if idx < data.shape[0]:
        reconstructed_full[idx] = reconstructed_windows[i, half_win]

# ---------------------------
# (7) 繪圖：原始數據 vs. 重構數據 (shape: (#Time, #Series))
plt.figure(figsize=(12, 8))
for i in range(data.shape[1]):
    plt.subplot(data.shape[1], 1, i+1)
    plt.plot(data[:, i], label='原始數據')
    plt.plot(reconstructed_full[:, i], label='重構數據', linestyle='--')
    plt.title(f'序列 {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()
