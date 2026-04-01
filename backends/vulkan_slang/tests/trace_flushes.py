"""Trace what triggers flush_stream calls during forward pass."""
import torch
import torch.nn.functional as F
import torch_vulkan
import traceback
import ctypes

device = torch.device("vulkan")
_c = torch_vulkan._c_ext

B, S, D, V = 2, 64, 256, 4096
n_heads, n_kv_heads, n_layers = 8, 4, 4

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return torch_vulkan.rms_norm(x, self.weight, self.eps)

class MiniQwen3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V, D)
        head_dim = D // n_heads
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(torch.nn.ModuleDict({
                "ln1": RMSNorm(D),
                "ln2": RMSNorm(D),
                "q": torch.nn.Linear(D, n_heads * head_dim, bias=False),
                "k": torch.nn.Linear(D, n_kv_heads * head_dim, bias=False),
                "v": torch.nn.Linear(D, n_kv_heads * head_dim, bias=False),
                "o": torch.nn.Linear(n_heads * head_dim, D, bias=False),
                "gate": torch.nn.Linear(D, D * 3, bias=False),
                "up": torch.nn.Linear(D, D * 3, bias=False),
                "down": torch.nn.Linear(D * 3, D, bias=False),
            }))
        self.norm_f = RMSNorm(D)
        self.lm_head = torch.nn.Linear(D, V, bias=False)
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

    def forward(self, x):
        h = self.embed(x)
        for L in self.layers:
            hn = L["ln1"](h)
            q = L["q"](hn).view(B, S, n_heads, self.head_dim).transpose(1, 2)
            k = L["k"](hn).view(B, S, n_kv_heads, self.head_dim).transpose(1, 2)
            v = L["v"](hn).view(B, S, n_kv_heads, self.head_dim).transpose(1, 2)
            if self.n_rep > 1:
                k = k.unsqueeze(2).expand(B, n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, n_heads, S, self.head_dim)
                v = v.unsqueeze(2).expand(B, n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, n_heads, S, self.head_dim)
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.full((S, S), float('-inf'), device=h.device), diagonal=1)
            attn = F.softmax(attn + causal, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
            h = h + L["o"](out)
            hn2 = L["ln2"](h)
            gate_up = torch_vulkan.swiglu(L["gate"](hn2), L["up"](hn2))
            h = h + L["down"](gate_up)
        h = self.norm_f(h)
        return self.lm_head(h)

model = MiniQwen3().to(device)
input_ids = torch.randint(0, V, (B, S)).to(device)
targets = torch.randint(0, V, (B, S)).view(-1).to(device)

# Warmup
logits = model(input_ids)
loss = F.cross_entropy(logits.view(-1, V), targets)
loss.backward()

# Now trace individual operations
print("\n=== Tracing Forward (1 layer) ===")

# Step through operations manually
def check_flush(label):
    f = _c._get_flush_count()
    d = _c._get_dispatch_count()
    pr = _c._get_preread_flush_count()
    print(f"  {label:<40} dispatches={d:>4}  flushes={f:>3}  preread={pr:>3}")

_c._reset_perf_counters()
check_flush("start")

h = model.embed(input_ids)
check_flush("embed")

L = model.layers[0]
hn = L["ln1"](h)
check_flush("rms_norm_1")

q = L["q"](hn)
check_flush("q_proj (linear)")

q = q.view(B, S, n_heads, model.head_dim).transpose(1, 2)
check_flush("q view+transpose")

k = L["k"](hn).view(B, S, n_kv_heads, model.head_dim).transpose(1, 2)
check_flush("k_proj + view+transpose")

v = L["v"](hn).view(B, S, n_kv_heads, model.head_dim).transpose(1, 2)
check_flush("v_proj + view+transpose")

k = k.unsqueeze(2).expand(B, n_kv_heads, model.n_rep, S, model.head_dim).reshape(B, n_heads, S, model.head_dim)
check_flush("k GQA expand")

v = v.unsqueeze(2).expand(B, n_kv_heads, model.n_rep, S, model.head_dim).reshape(B, n_heads, S, model.head_dim)
check_flush("v GQA expand")

scale = model.head_dim ** -0.5
attn = (q @ k.transpose(-2, -1)) * scale
check_flush("q @ k.T * scale")

causal = torch.triu(torch.full((S, S), float('-inf'), device=h.device), diagonal=1)
check_flush("causal mask")

attn = attn + causal
check_flush("attn + causal")

attn = F.softmax(attn, dim=-1)
check_flush("softmax")

out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
check_flush("attn @ v + reshape")

out = L["o"](out)
check_flush("o_proj")

h = h + out
check_flush("residual add 1")

hn2 = L["ln2"](h)
check_flush("rms_norm_2")

gate_out = L["gate"](hn2)
check_flush("gate_proj")

up_out = L["up"](hn2)
check_flush("up_proj")

gate_up = torch_vulkan.swiglu(gate_out, up_out)
check_flush("swiglu")

down = L["down"](gate_up)
check_flush("down_proj")

h = h + down
check_flush("residual add 2")

h = model.norm_f(h)
check_flush("final rms_norm")

logits = model.lm_head(h)
check_flush("lm_head")

loss = F.cross_entropy(logits.view(-1, V), targets)
check_flush("cross_entropy")

print(f"\nTotal: {_c._get_dispatch_count()} dispatches, {_c._get_flush_count()} flushes")
