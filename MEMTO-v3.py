import os
import tqdm
import math
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def shape_print(arr):
    print(tuple(arr.shape))

N_BATCH = 128
LENGTH = 60
N_SERIES = 3

arr = torch.randn(N_BATCH, LENGTH, N_SERIES)
print("(N_BATCH, LENGTH, N_SERIES) =", end = " ")
shape_print(arr)

class Shape_Check(nn.Module):
    def __init__(self, length, n_series):
        super(Shape_Check, self).__init__()
        self.length, self.n_series = length, n_series
    
    def forward(self, x):
        if x.shape[1:] != (self.length, self.n_series):
            word = "Shape does not fit! It shall be (N_BATCH, {}, {}) ".format(self.length, self.n_series)
            word += "But get ()".format(tuple(x.shape))
            raise RuntimeError(word)
        return x

class Permute(nn.Module):
    def __init__(self, *n_dim):
        super(Permute, self).__init__()
        self.dims = n_dim
        
    def forward(self, x):
        return x.permute(self.dims)
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_head, n_layer):
        super(Encoder, self).__init__()
        enc = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layer)
    def forward(self, x):
        return self.enc(x)
    
class MatMul(nn.Module):
    def __init__(self, y):
        super(MatMul, self).__init__()
        self.y = y
    def forward(self, x):
        return x@self.y
    
class Memory_Attention(nn.Module):
    def __init__(self):
        super(Memory_Attention, self).__init__()
    def forward(self, Tuple):
        qs, memory = Tuple
        s = qs@memory.T
        s = s.permute(0, 2, 1)
        v = nn.Softmax(dim = 2)(s)
        r = v@qs
        return r, memory
    
class U_W_psi(nn.Module):
    def __init__(self, d_model):
        super(U_W_psi, self).__init__()
        self.U_psi = nn.Linear(d_model, d_model, bias = False)
        self.W_psi = nn.Linear(d_model, d_model, bias = False)
    def forward(self, Tuple):
        r, memory = Tuple
        r_mean = r.mean(dim = 0)
        psi = self.U_psi(memory) + self.W_psi(r_mean)
        return (1 - psi)*memory + psi*r_mean
    
class Retrieve(nn.Module):
    def __init__(self):
        super(Retrieve, self).__init__()
    def forward(self, Tuple):
        qs, memory = Tuple
        rr = qs@memory.T
        rr = nn.Softmax(dim = 2)(rr)
        q_telta = rr@memory
        q_hat = torch.cat([qs, q_telta], dim = -1)
        return q_hat, rr

from sklearn.cluster import KMeans

class MEMTO(nn.Module):
    def __init__(self, length, n_series, d_model, n_head, n_layer, n_memory, x, device = "cuda"):
        super(MEMTO, self).__init__()
        # Encoder Block
        self.encoder = nn.Sequential(
            Shape_Check(length, n_series), # (N_BATCH, LENGTH, N_SERIES)
            nn.Linear(n_series, d_model), # Embedded to (N_BATCH, LENGTH, D_MODEL)
            Permute(1, 0, 2), # Permute to (LENGTH, N_BATCH, D_MODEL)
            Encoder(d_model, n_head, n_layer), # Encode to (LENGTH, N_BATCH, D_MODEL)
            Permute(1, 0, 2) # Permute to (N_BATCH, LENGTH, D_MODEL)
        )
        
        # Memory init
        q = self.encoder(x).reshape(-1, d_model).detach().numpy()
        self.memory = nn.Parameter(
            torch.tensor(
                KMeans(n_clusters=n_memory, random_state=42).fit(q).cluster_centers_, dtype=torch.float) # (N_MEMORY, D_MODEL)
        ).to(device)
        
        # Memory update gate
        self.memory_gate = nn.Sequential(
            Memory_Attention(), # Attention from qs, memory to (N_BATCH, N_MEMORY, LENGTH), memory
            U_W_psi(d_model) # New memory still (N_MEMORY, D_MODEL)
        )
        
        # Memory retrieve gate
        self.memory_retrieve = nn.Sequential(
            Retrieve() # Retrieve to (N_BATCH, LENGTH, 2*D_MODEL)
        )
        
        # Decoder Block
        self.decoder = nn.Sequential(
            nn.Linear(2*d_model, d_model), # Dense to (N_BATCH, LENGTH, D_MODEL)
            nn.Linear(d_model, n_series) # Dense to (N_BATCH, LENGTH, N_SERIES)
        )
        self.device = device
        self.to(device)
    
    def forward(self, x):
        qs = self.encoder(x)
        new_memory = self.memory_gate((qs, self.memory))
        q, _ = self.memory_retrieve((qs, new_memory))
        recon = self.decoder(q)
        return recon
        
    def fit(self, dataloader, epochs):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr = 5e-5)
        for epoch in range(1, epochs + 1):
            num_loss = 0.0
            for data in dataloader:
                # Inference
                data = data.to(self.device)
                qs = self.encoder(data)
                new_memory = self.memory_gate((qs, self.memory))
                q, rr = self.memory_retrieve((qs, new_memory))
                recon = self.decoder(q)
                
                loss = nn.MSELoss()(data, recon)
                loss += .01*torch.mean(torch.sum((torch.sum(-rr*torch.log(rr+1e-8), dim = -1)), dim = -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_loss += loss.item()
            print("[Epoch {}] Train-Loss={}".format(epoch, round(num_loss, 4)), end = "\r")
            
D_MODEL = 32
N_HEAD = 8
N_MEMORY = 16
N_LAYER = 6
    
    
model = MEMTO(length = LENGTH, 
              n_series = N_SERIES, 
              d_model = D_MODEL, 
              n_head = N_HEAD, 
              n_layer = N_LAYER, 
              n_memory = N_MEMORY,
              x = arr,
              device = "cuda"
             )
model.to(device)
# model(arr).shape
tr_loader = DataLoader(arr, batch_size=32, shuffle=True)
model.fit(tr_loader, 100)
