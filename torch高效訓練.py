import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms # 假設是圖片數據，可替換
from tqdm import tqdm
import os
import time

# --- 1. 超參數與設定 (Hyperparameters & Settings) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# !! 重要：根據你的 CPU 核心數和實驗調整 !!
# 通常設置為實體核心數，或稍多一些，但過多可能導致競爭和開銷
NUM_WORKERS = min(os.cpu_count(), 16) # 示例值，A100 通常搭配較多核心 CPU

# !! 重要：根據你的 A100 VRAM 大小和模型大小調整 !!
# 盡量設大，直到出現 Out-of-Memory (OOM) 錯誤前
BATCH_SIZE = 512 # 示例值，A100 可以嘗試 1024 或更大

LEARNING_RATE = 1e-3
EPOCHS = 50 # 根據你的需求調整
LATENT_DIM = 64 # VAE 潛在空間維度
INPUT_DIM = 28 * 28 # 假設是 MNIST 類型數據 (784)
H_DIM = 256 # 隱藏層維度

# --- 2. 數據準備 (Data Preparation) ---
# 創建一個 Dummy Dataset，你需要替換成你真實的數據加載方式
# 假設你的數據是存儲在某個地方的 Tensor 或 Numpy array
class MyLargeDataset(Dataset):
    def __init__(self, num_samples=1_000_000, data_dim=INPUT_DIM, transform=None):
        print("Initializing dummy dataset (replace with your actual data loading)...")
        # 在實際應用中，這裡應該是讀取你數據的路徑或引用
        # 為了演示，我們創建隨機數據 (非常耗內存，實際不要這樣做)
        # self.data = torch.randn(num_samples, data_dim)
        # 更好的演示方式是不直接創建所有數據，而是在 __getitem__ 中生成或讀取
        self.num_samples = num_samples
        self.data_dim = data_dim
        self.transform = transform
        print("Dummy dataset initialized.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 在實際應用中，這裡會根據 idx 從磁盤或其他來源加載數據
        # 例如: img = Image.open(self.file_list[idx]).convert('L')
        #       data = self.transform(img)
        # 為了演示，生成隨機數據
        sample = torch.randn(self.data_dim) # 模擬加載一個樣本
        if self.transform:
            # 假設 transform 是針對 tensor 的 (例如，如果數據已是 Tensor)
            # 如果是圖像，transform 通常在加載後應用
            pass # 在這裡應用你的 transform
        return sample

# 數據預處理/轉換 (根據你的數據調整)
transform = transforms.Compose([
    # transforms.ToTensor(), # 如果是 PIL Image
    # transforms.Normalize((0.5,), (0.5,)) # 如果需要標準化
    # 在這裡添加你的數據預處理步驟
])

print("Loading dataset...")
# 使用你的真實 Dataset 替換 MyLargeDataset
dataset = MyLargeDataset(num_samples=1_000_000, data_dim=INPUT_DIM, transform=transform)
print("Dataset loaded.")

print("Creating DataLoader...")
# 高效的 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,    # <--- 利用多核 CPU
    pin_memory=True,        # <--- 加速 CPU to GPU 傳輸
    persistent_workers=True if NUM_WORKERS > 0 else False, # <--- 減少 worker 重啟開銷 (PyTorch 1.9+)
    drop_last=True          # 在大數據集上通常建議 True，避免最後一個 batch 過小影響統計
)
print("DataLoader created.")

# --- 3. VAE 模型定義 (VAE Model Definition) ---
class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim) # 均值
        self.fc_logvar = nn.Linear(h_dim, z_dim) # 對數方差
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 從標準正態分佈採樣
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        # 使用 Sigmoid 輸出確保像素值在 [0, 1] 或根據你的數據範圍調整
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# --- 4. 損失函數 (Loss Function) ---
def loss_function(recon_x, x, mu, logvar):
    # 重構損失 (Reconstruction Loss) - 使用二元交叉熵或均方誤差
    # 假設輸入數據已標準化到 [0, 1]
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')
    # MSE = F.mse_loss(recon_x, x.view(-1, INPUT_DIM), reduction='sum')

    # KL 散度 (KL Divergence) - 計算潛在空間分佈與標準正態分佈的差異
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# --- 5. 初始化模型、優化器、混合精度 ---
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 使用 torch.compile (PyTorch 2.0+) 進一步優化 (可選但推薦)
# mode 可以是 'default', 'reduce-overhead', 'max-autotune'
try:
    model = torch.compile(model, mode='max-autotune')
    print("Model compiled successfully (PyTorch 2.0+).")
except Exception as e:
    print(f"torch.compile failed (likely using PyTorch < 2.0 or encountering an issue): {e}")


# 初始化混合精度 GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()) # 僅在 CUDA 可用時啟用

# --- 6. 訓練循環 (Training Loop) ---
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    start_time = time.time()
    # 使用 tqdm 顯示進度條
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, data in pbar:
        # data 已經在 DataLoader 的 worker 中被處理 (如果定義了 transform)
        # 將數據移至 GPU，使用 non_blocking=True 配合 pin_memory
        data = data.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # 使用 autocast 上下文管理器進行混合精度前向傳播和損失計算
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)

        # 使用 scaler 縮放損失並執行反向傳播
        scaler.scale(loss).backward()

        # 使用 scaler 更新模型參數
        scaler.step(optimizer)

        # 更新 scaler 的縮放因子
        scaler.update()

        epoch_loss += loss.item()

        # 更新進度條顯示
        if batch_idx % 50 == 0: # 每 50 個 batch 更新一次 loss 顯示
             pbar.set_postfix({'Loss': loss.item() / len(data)})

    end_time = time.time()
    avg_loss = epoch_loss / len(train_loader.dataset)
    epoch_time = end_time - start_time
    print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} Time: {epoch_time:.2f}s')

print("Training finished.")

# --- 7. 保存模型 (Save Model) ---
# 建議保存 state_dict
torch.save(model.state_dict(), 'vae_model_state_dict.pth')
# 如果使用了 torch.compile， 加載時需要先創建未 compile 的模型實例，然後 load_state_dict
# model_new = VAE().to(DEVICE)
# model_new.load_state_dict(torch.load('vae_model_state_dict.pth'))
# model_compiled_loaded = torch.compile(model_new) # 如果需要繼續使用 compile 的模型

print("Model saved.")
