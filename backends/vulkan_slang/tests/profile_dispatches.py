"""Profile dispatch and flush counts during a Qwen training step."""
import torch
import torch.nn.functional as F
import torch_vulkan
import time

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
        # First ln1 is a plain rms_norm (no preceding add to fuse)
        hn = torch_vulkan.rms_norm(h, self.layers[0]["ln1"].weight, 1e-6)
        for i, L in enumerate(self.layers):
            q = L["q"](hn).view(B, S, n_heads, self.head_dim).transpose(1, 2)
            k = L["k"](hn).view(B, S, n_kv_heads, self.head_dim).transpose(1, 2)
            v = L["v"](hn).view(B, S, n_kv_heads, self.head_dim).transpose(1, 2)
            scale = self.head_dim ** -0.5
            # Flash attention: handles GQA natively (K/V stay unexpanded [B, n_kv_heads, S, D])
            # Saves expand+reshape dispatches and eliminates [B*H, N, S] intermediate matrix
            attn_out = torch_vulkan.flash_attention(q, k, v, scale, is_causal=True)
            out = attn_out.transpose(1, 2).reshape(B, S, -1)
            # Fuse: add(h, attn_out) + ln2 → hn2, h_new in one dispatch
            hn2, h = torch_vulkan.add_rms_norm(h, L["o"](out), L["ln2"].weight)
            gate_up = torch_vulkan.swiglu(L["gate"](hn2), L["up"](hn2))
            # Fuse: add(h, mlp_out) + ln1_next → hn_next, h_new in one dispatch
            if i < n_layers - 1:
                hn, h = torch_vulkan.add_rms_norm(h, L["down"](gate_up), self.layers[i+1]["ln1"].weight)
            else:
                h = h + L["down"](gate_up)
        h = self.norm_f(h)
        return self.lm_head(h)

model = MiniQwen3().to(device)
optimizer = torch_vulkan.SGD(model.parameters(), lr=0.01)
input_ids = torch.randint(0, V, (B, S), dtype=torch.int32).to(device)  # int32 saves i64→i32 dispatch per embedding
targets = torch.randint(0, V, (B, S)).view(-1).to(torch.int32).to(device)  # int32 saves 2 i64→i32 dispatches per CE step

# Warmup
for _ in range(3):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, V), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    _c._flush()  # Ensure clean state between warmup iterations

# Profile each phase — flush before each phase to avoid timing contamination
def profile_phase(name, fn):
    _c._flush()  # Flush accumulated GPU work from previous phases
    _c._reset_perf_counters()
    t0 = time.perf_counter()
    result = fn()
    _c._flush()  # Flush to get accurate GPU completion time
    t1 = time.perf_counter()
    d = _c._get_dispatch_count()
    f = _c._get_flush_count()
    w = _c._get_war_flush_count()
    pr = _c._get_preread_flush_count()
    cap = _c._get_capacity_flush_count()
    dp = _c._get_descpool_flush_count()
    b_emitted = _c._get_barrier_count()
    b_skipped = _c._get_barrier_skip_count()
    ms = (t1 - t0) * 1000
    skip_pct = b_skipped/(b_emitted+b_skipped)*100 if (b_emitted+b_skipped) > 0 else 0
    print(f"  {name:<20} {ms:>8.2f} ms  dispatches={d:>4}  flushes={f:>3} (preread={pr} cap={cap} descpool={dp} WAR={w})  barriers={b_emitted}/{b_emitted+b_skipped} ({skip_pct:.0f}% skipped)  avg={ms/max(d,1):.3f} ms/dispatch")
    return result

print(f"\nConfig: B={B}, S={S}, D={D}, L={n_layers}, V={V}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

optimizer.zero_grad()
logits = profile_phase("Forward", lambda: model(input_ids))
loss = profile_phase("CE + to_cpu", lambda: F.cross_entropy(logits.view(-1, V), targets))
profile_phase("Backward", lambda: loss.backward())
profile_phase("Optimizer step", lambda: optimizer.step())
profile_phase("Zero grad", lambda: optimizer.zero_grad())

# Also profile just an mm to get single-dispatch overhead
print("\n--- Single-op overhead ---")
a = torch.randn(256, 256, device=device)
b = torch.randn(256, 256, device=device)
# Warmup
_ = a @ b
# Timed
for _ in range(3):
    _c._reset_perf_counters()
    t0 = time.perf_counter()
    c = a @ b
    t1 = time.perf_counter()
    d = _c._get_dispatch_count()
    f = _c._get_flush_count()
    print(f"  mm(256,256)         {(t1-t0)*1000:.3f} ms  dispatches={d}  flushes={f}")

# And a .cpu() readback
_c._reset_perf_counters()
t0 = time.perf_counter()
_ = c.cpu()
t1 = time.perf_counter()
d = _c._get_dispatch_count()
f = _c._get_flush_count()
print(f"  .cpu() readback     {(t1-t0)*1000:.3f} ms  dispatches={d}  flushes={f}")
