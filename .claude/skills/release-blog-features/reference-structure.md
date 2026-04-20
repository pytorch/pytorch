# PyTorch Release Blog — Output Template & Examples

Use this template when drafting a release blog. It mirrors the structure
of recent posts on pytorch.org/blog (2.9, 2.10, 2.11).

## Template

```markdown
# PyTorch <VERSION> Release Blog

We are excited to announce the release of PyTorch® <VERSION>
([release notes](https://github.com/pytorch/pytorch/releases/tag/v<VERSION>.0))!
This release features <N> commits from <M> contributors since PyTorch
<PREV_VERSION>.

## Highlights

- <Feature 1 — one line>
- <Feature 2 — one line>
- <Feature 3 — one line>
- ...

Along with <VERSION>, we are also releasing updates to the PyTorch
domain libraries. See the blog posts for [TorchAudio](#), [TorchVision](#),
and [ExecuTorch](#).

---

## Beta Features

### [Beta] <Feature name> — <component>

<1-3 sentences describing what it does, who should care, and what
changed vs. the previous release.>

See [#<PR>](https://github.com/pytorch/pytorch/pull/<PR>) and the
[documentation](<link>).

### [Beta] <Feature name> — <component>
...

---

## Prototype Features

### [Prototype] <Feature name> — <component>
<description>

### [Prototype] <Feature name> — <component>
...

---

## Performance Improvements

### <component> — <short title>
<description, with any benchmark numbers lifted directly from the PR>

---

## Deprecations and Backwards-Incompatible Changes

- <Change + migration path>. See [#<PR>](...).

---

## Non-Feature Updates

- **CUDA default**: <old> → <new>
- **Python support**: <range>
- **Release cadence**: <note if it changed>
```

## Example (filled in, abbreviated — based on the 2.11 blog)

```markdown
# PyTorch 2.11 Release Blog

We are excited to announce the release of PyTorch® 2.11
([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0))!
This release features 2,723 commits from 432 contributors since PyTorch 2.10.

## Highlights

- Differentiable collectives enable backprop through distributed ops.
- FlexAttention gets a FlashAttention-4 backend on Hopper/Blackwell
  with 1.2×-3.2× speedups over Triton.
- MPS adds async error reporting and new distributions.
- RNN/LSTM export to GPU via `torch.export`.
- ROCm gains device-side asserts and TopK wins.
- XPU adds CUDA-graph-style capture/replay.
- FP16 half-precision GEMM on CPU via OpenBLAS.
- CUDA 13 is now the default build; TorchScript is deprecated.

## API-Unstable Features

### [API-Unstable] Differentiable Collectives for Distributed Training

Collective operations (all_reduce, all_gather, reduce_scatter) now
support backpropagation, enabling new distributed-training research
patterns that previously required manual gradient plumbing.

See [#<TODO>] and the [distributed docs](...).

### [API-Unstable] FlexAttention with FlashAttention-4 Backend

FlexAttention now targets FlashAttention-4 on Hopper and Blackwell GPUs,
with auto-generated CuTeDSL score/mask functions. Reported speedups are
1.2×-3.2× over the Triton backend.

...

## Non-Feature Updates

- **CUDA default**: CUDA 12.x → CUDA 13 (x86_64 and ARM). CPU-only and
  CUDA 12.8 builds remain available.
- **TorchScript**: deprecated since 2.10 — migrate to `torch.export` or
  ExecuTorch.
- **Release cadence**: moving to one release every 2 months in 2026.
```

## Stability tags — how to pick

| Tag                | When to use                                                                  |
|--------------------|------------------------------------------------------------------------------|
| `[Stable]`         | API is public, documented, covered by BC policy. Rarely tagged explicitly.   |
| `[Beta]`           | Public API, usable in production, minor signature changes still possible.    |
| `[Prototype]`      | Public API, but the team reserves the right to remove or redesign it.        |
| `[API-Unstable]`   | Equivalent to Prototype in recent (2.9+) blogs — use this for consistency.   |

When in doubt, match the tag the PR author used in the PR body. If the
PR body has no tag, default to `[Prototype]` for new features and flag it
for the release manager.

## Component names — canonical list

Use these exactly when they apply (matches recent blog posts):

- Dynamo / torch.compile
- Inductor
- Distributed
- Export / AOTInductor
- Quantization
- Profiler
- ONNX
- NN / Frontend
- MPS
- ROCm
- XPU
- CPU / Arm
- CUDA
- libtorch / C++ ABI
- Release Engineering

## Things to avoid

- **Don't invent speedup numbers.** Only cite numbers that appear in the
  PR body or on a linked dashboard.
- **Don't list ghstack sub-PRs.** Collapse a stack to one entry.
- **Don't include reverts.** If `#A` was reverted by `#B` before the
  branch cut, neither belongs in the blog.
- **Don't include internal refactors.** If a PR only moves code around
  without changing a user-visible API or perf number, skip it.
- **Don't write commit-style descriptions.** "Fix segfault in foo when
  bar" belongs in release notes, not the blog.
