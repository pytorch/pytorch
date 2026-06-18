# Capture recipe

Author a throwaway capture script in `agent_space/` that wraps the exact step
being compared in `DebugMode` and writes `dm.debug_string()` to a text file.
Do not build a separate log format unless the plain dump is too large to inspect.

The useful DebugMode settings are:

- `record_realtensor=True` — record real tensor ops.
- `record_nn_module=True` — add `nn.Module` enter markers from the module
  tracker.
- `record_ids=True` — give tensors stable IDs in the dump, so input/output
  flow is easier to follow.
- `record_stack_trace=True` — include source summaries with
  `debug_string(show_stack_trace=True)`. Turn this off if the dump is too noisy.
- `record_profiler_context=True` — keep profiler/record_function context
  markers. This is the default, but pass it explicitly in the recipe.
- `run_compile_with_interpreter=True` — use when debugging an aot/eager compiled
  region and you need FX node/module metadata in the dump.
- `DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True)` — add output
  hashes and pre-op input hashes to each op line.

`log_tensor_hashes` is the key numerics signal. The output `hash` tells you
whether this op produced the same value in both runs. `input_hash` is computed
before the op executes, so if an output hash differs while the input hashes
match, that op is the likely source. If the input hashes already differ, the
op is only carrying an upstream divergence.

## Step 1: probe the dump shape

`DebugMode` is an internal debugging tool, so first run a tiny probe in the
same checkout/environment. Confirm that module markers and tensor hashes show
up in the plain string dump before adapting the recipe to the real model:

```python
import torch
import torch.nn as nn
from torch.utils._debug_mode import DebugMode

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(self.proj(x))

m = nn.Sequential(Block(), Block())
dm = DebugMode(
    record_realtensor=True,
    record_nn_module=True,
    record_ids=True,
    record_stack_trace=True,
    record_profiler_context=True,
)
with dm, DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True):
    m(torch.randn(8, 64)).sum().backward()

print(dm.debug_string(show_stack_trace=True))
```

Look for:

- `[nn.Mod] ...` lines around module execution.
- Tensor IDs like `$0`, `$1`, etc. on op inputs/outputs.
- Per-op hash comments containing `hash` and `input_hash`.
- Useful source summaries when `show_stack_trace=True`.

If `record_nn_module=True` does not give useful layer names for the target
model, keep the dump generic: use stack traces, profiler scopes, tensor IDs, and
nearby op order. Do not monkey-patch modules just to manufacture layer names.

## Step 2: capture the real step

Use this as the starting point for each side of the comparison. The only output
is DebugMode's own text dump.

```python
import torch
from torch.utils._debug_mode import DebugMode

def capture_debug_string(
    model,
    run_fn,
    path,
    *,
    show_stack_trace=True,
    run_compile_with_interpreter=False,
):
    dm = DebugMode(
        record_realtensor=True,
        record_nn_module=True,
        record_ids=True,
        record_stack_trace=show_stack_trace,
        record_profiler_context=True,
        run_compile_with_interpreter=run_compile_with_interpreter,
    )
    with dm, DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True):
        run_fn(model)

    with open(path, "w") as f:
        f.write(dm.debug_string(show_stack_trace=show_stack_trace))
        f.write("\n")
```

Drive it once per run, with identical setup on both sides:

```python
def build():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Net().to(device), torch.randn(16, 256, device=device)

def step(m, x):
    # For inference-only comparisons, return or consume m(x) without backward.
    m(x).sum().backward()

m, x = build()
capture_debug_string(m, lambda mm: step(mm, x), "run_A_debug.txt")

# Change exactly one thing under test: compile, a distributed wrapper, a flag,
# a refactor, etc.
m, x = build()
capture_debug_string(m, lambda mm: step(mm, x), "run_B_debug.txt")
```

For strict bitwise checks, switch from the default `norm` hash to:

```python
DebugMode.log_tensor_hashes(hash_fn="hash_tensor", hash_inputs=True)
```

or log both:

```python
DebugMode.log_tensor_hashes(hash_fn=["norm", "hash_tensor"], hash_inputs=True)
```

For `torch.compile`/AOT eager comparisons, try:

```python
capture_debug_string(
    m,
    lambda mm: step(mm, x),
    "run_A_debug.txt",
    run_compile_with_interpreter=True,
)
```

## Single-step training check

Before collecting a huge dump from a training job, compare scalar outputs from
one deterministic step. This tells you where to spend attention:

```python
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().double().pow(2).sum()
    return total.sqrt()

def training_step(model, batch):
    loss = model(batch).sum()
    loss.backward()
    return loss.detach(), grad_norm(model).detach()
```

- Loss diverges: start with forward hashes.
- Loss matches bitwise but grad norm diverges: forward is probably fine; start
  at the first backward hash mismatch.
- Loss and grad norm match: look at optimizer state, data loading, RNG
  consumption after the checked step, or later iterations.

## Example dump snippets

These excerpts were generated from a tiny two-layer CPU model with
`conda activate pytorch-3.10` on torch `2.14.0a0+gitc45c69c`.

Eager dumps have source summaries, module scopes, tensor IDs, and hash comments:

```text
    # File: debugmode_dump_smoke.py:49 in capture, code: loss = model(x)
    [nn.Mod] Net

      # File: debugmode_dump_smoke.py:27 in forward, code: return self.features(x).sum()
      [nn.Mod] Net.features
        [nn.Mod] Net.features.0

          # File: debugmode_dump_smoke.py:18 in forward, code: return torch.relu(self.proj(x))
          [nn.Mod] Net.features.0.proj
            aten::t(t$0: f32[4, 4])  ->  t$1: f32[4, 4]  # {'input_hash': ((3.243980646133423,), {}), 'hash': 3.243980646133423}
            aten::addmm(t$2: f32[4], t$3: f32[2, 4], t$1: f32[4, 4])  ->  t$4: f32[2, 4]  # {'input_hash': ((1.0543809533119202, 7.07875682413578, 3.243980646133423), {}), 'hash': 4.30271789431572}
          aten::relu(t$4: f32[2, 4])  ->  t$5: f32[2, 4]  # {'input_hash': ((4.30271789431572,), {}), 'hash': 2.3040474951267242}

      aten::sum(t$10: f32[2, 4])  ->  t$11: f32[]  # {'input_hash': ((1.0529326498508453,), {}), 'hash': 1.052932620048523}
```

With `run_compile_with_interpreter=True`, compiled/AOT dumps may include compile
region annotations and FX-derived source/module context. Names may be rooted at
Dynamo locals like `L['self']`:

```text
  [aot_eager region (compile)] enter

    # File: debugmode_dump_smoke.py:26 in forward, code: def forward(self, x: torch.Tensor) -> torch.Tensor:
    [nn.Mod (compile)] L['self'].features
      [nn.Mod (compile)] L['self'].features.0
        [nn.Mod (compile)] L['self'].features.0.proj

          # File: debugmode_dump_smoke.py:18 in forward, code: return torch.relu(self.proj(x))
          aten::t(t$0: f32[4, 4])  ->  t$1: f32[4, 4]  # {'input_hash': ((3.243980646133423,), {}), 'hash': 3.243980646133423}
          aten::addmm(t$2: f32[4], t$3: f32[2, 4], t$1: f32[4, 4])  ->  t$4: f32[2, 4]  # {'input_hash': ((1.0543809533119202, 7.07875682413578, 3.243980646133423), {}), 'hash': 4.30271789431572}
        aten::relu(t$4: f32[2, 4])  ->  t$5: f32[2, 4]  # {'input_hash': ((4.30271789431572,), {}), 'hash': 2.3040474951267242}

    aten::sum(t$10: f32[2, 4])  ->  t$12: f32[]  # {'input_hash': ((1.0529326498508453,), {}), 'hash': 1.052932620048523}
  [aot_eager region (compile)] exit
```

The exact operator names may differ (`linear` in eager can appear as `addmm`
after tracing/lowering). Match by scope, shape, tensor flow, and hashes rather
than expecting identical text.

When a run is intentionally perturbed, the first forward mismatch is visible in
the hash comments. In this example the second layer's weight changed, so the
first different line has a matching activation input hash but different weight
and output hashes:

```text
# Run A
aten::addmm(..., t$5: f32[2, 4], t$7: f32[4, 4])  ->  t$9: f32[2, 4]  # {'input_hash': ((1.1871147155761719, 2.3040474951267242, 4.564039468765259), {}), 'hash': 1.5524791777133942}

# Run B
aten::addmm(..., t$5: f32[2, 4], t$7: f32[4, 4])  ->  t$9: f32[2, 4]  # {'input_hash': ((1.1871147155761719, 2.3040474951267242, 4.689039468765259), {}), 'hash': 1.7220753133296967}
```

## Reading the dump

The dump is indented by call depth. Module markers, profiler scopes, operator
calls, tensor IDs, and hash comments are all in one text artifact. Compare
`run_A_debug.txt` and `run_B_debug.txt` directly with a text diff and inspect
the first op where the output `hash` differs.

When an output hash differs, compare that op's `input_hash` values:

- Inputs match and output differs -> this op is the likely root cause.
- Inputs already differ -> the divergence is upstream; walk backward through
  tensor IDs and earlier hash comments.
- Every op differs from the first real op -> setup mismatch, usually dtype,
  seed, input, RNG, or determinism.
- Tiny last-digit differences that do not propagate are usually reduction-order
  noise. Use judgment; the `norm` hash is a float64 reduction, not a proof of
  bit-exact equality.
