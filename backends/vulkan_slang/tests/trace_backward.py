"""Trace what triggers flush_stream calls during backward pass."""
import torch
import torch.nn.functional as F
import torch_vulkan

device = torch.device("vulkan")
_c = torch_vulkan._c_ext

B, S, D, V = 2, 64, 256, 4096
n_heads, n_kv_heads, n_layers = 8, 4, 1  # Use 1 layer for simpler tracing
head_dim = D // n_heads
n_rep = n_heads // n_kv_heads

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

    def forward(self, x):
        h = self.embed(x)
        for L in self.layers:
            hn = L["ln1"](h)
            q = L["q"](hn).view(B, S, n_heads, head_dim).transpose(1, 2)
            k = L["k"](hn).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
            v = L["v"](hn).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
            if n_rep > 1:
                k = k.unsqueeze(2).expand(B, n_kv_heads, n_rep, S, head_dim).reshape(B, n_heads, S, head_dim)
                v = v.unsqueeze(2).expand(B, n_kv_heads, n_rep, S, head_dim).reshape(B, n_heads, S, head_dim)
            scale = head_dim ** -0.5
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

model.zero_grad()

# Now trace backward with hooks on every parameter and module
print("=== Tracing Backward (1 layer) ===\n")

# Register hooks on all parameters
param_hooks = []
for name, param in model.named_parameters():
    def make_hook(n):
        def hook(grad):
            d = _c._get_dispatch_count()
            f = _c._get_flush_count()
            pr = _c._get_preread_flush_count()
            war = _c._get_war_flush_count()
            print(f"  grad({n:<35}) d={d:>4} f={f:>3} pr={pr:>3} war={war:>3}")
            return grad
        return hook
    h = param.register_hook(make_hook(name))
    param_hooks.append(h)

# Register hooks on module outputs (backward hooks)
module_hooks = []
for name, mod in model.named_modules():
    if name == '': continue
    def make_bwd_hook(n):
        def hook(module, grad_input, grad_output):
            d = _c._get_dispatch_count()
            f = _c._get_flush_count()
            pr = _c._get_preread_flush_count()
            war = _c._get_war_flush_count()
            print(f"  bwd({n:<36}) d={d:>4} f={f:>3} pr={pr:>3} war={war:>3}")
        return hook
    h = mod.register_full_backward_hook(make_bwd_hook(name))
    module_hooks.append(h)

# Forward
_c._reset_perf_counters()
logits = model(input_ids)
print(f"After forward: d={_c._get_dispatch_count()}, f={_c._get_flush_count()}, pr={_c._get_preread_flush_count()}")

loss = F.cross_entropy(logits.view(-1, V), targets)
print(f"After CE:      d={_c._get_dispatch_count()}, f={_c._get_flush_count()}, pr={_c._get_preread_flush_count()}")

# Backward
loss.backward()
print(f"\nAfter backward: d={_c._get_dispatch_count()}, f={_c._get_flush_count()}, pr={_c._get_preread_flush_count()}")

# Cleanup
for h in param_hooks + module_hooks:
    h.remove()
