import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1

    self.n_head = config.n_head
    self.n_embd = config.n_embd

    self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)

    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
    y = y.transpose(1, 2).contiguous().view(B, T, C) 
    y = self.c_proj(y)
    return y


class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd,  4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x)) 
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024        
  vocab_size: int = 50257       
  n_layer: int = 12             
  n_head: int = 12              
  n_embd: int = 768             

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd), 
      wpe = nn.Embedding(config.block_size, config.n_embd), 
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
      ln_f = nn.LayerNorm(config.n_embd) 
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight
    self.apply(self._init_weights)

  
  def _init_weights(self, module):
    
    if isinstance(module, nn.Linear):
      std = 0.02

      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5 

      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def forward(self, idx, targets=None):
  
    B, T = idx.size()
    assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'

    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos) 
    tok_emb = self.transformer.wte(idx) 
    x = tok_emb + pos_emb 
    
    for block in self.transformer.h:
       x = block(x)

    
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    loss = None

    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
  
    return logits, loss

# --------------------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
   
  def __init__(self, B, T):
    self.B = B
    self.T = T

    with open('input.txt', 'r') as f:
      text = f.read()
  
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
    self.tokens = torch.tensor(tokens)
    print(f'loaded {len(self.tokens)} tokens')
    print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

   
    self.current_position = 0

  
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position: self.current_position + B * T + 1]
    x = buf[:-1].view(B, T) 
    y = buf[1:].view(B, T) 

    
    self.current_position += B * T

    if self.current_position + ( B * T + 1) > len(self.tokens):
      self.current_position = 0
    
    return x, y

# --------------------------------------------------------------------------------------------------------
import time

torch.manual_seed(1337)
device = 'cpu'

if torch.cuda.is_available():
   torch.cuda.manual_seed(1337)
   device = 'cuda'

elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
  torch.mps.manual_seed(1337)
  device = 'mps'

print(f'using device: {device}')

train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304, block_size=1024, n_layer=1, n_head=6, n_embd=6*64))
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(1):
  t0 = time.time()
  x, y = train_loader.next_batch()
  x, y = x.to(device), y.to(device)
  optimizer.zero_grad()

  with torch.autocast(device_type=device, dtype=torch.bfloat16):
   logits, loss = model(x, y)
  
  loss.backward()
  optimizer.step()
  # torch.mps.synchronize() # Waits for all kernels in all streams on a MPS device to complete.
  t1 = time.time()
  dt = (t1 - t0) * 1000
  tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
  print(f'step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}')

import sys
sys.exit(0)
