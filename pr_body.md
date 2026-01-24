## Summary

This PR adds native MPS support for Poisson and Dirichlet sampling (plus the underlying `_standard_gamma` kernel), unblocking **Cell2location** on Apple Silicon. This enables `torch.poisson`, `torch._sample_dirichlet`, and `torch._standard_gamma` to run fully on MPS without CPU fallback.

## Community Impact

### Cell2location
[Cell2location](https://github.com/BayraktarLab/cell2location) is a widely-used spatial transcriptomics tool built on Pyro/PyTorch. It relies on **NegativeBinomial** and **Dirichlet** sampling, which previously triggered MPS fallback because `aten::poisson` and `aten::_sample_dirichlet` were missing.

**Without this PR:** users must set `PYTORCH_ENABLE_MPS_FALLBACK=1`, causing major slowdowns.

**With this PR:** Cell2location models run fully on GPU for Apple Silicon.

Related issues:
- https://github.com/BayraktarLab/cell2location/issues/406
- https://github.com/BayraktarLab/cell2location/issues/221
- https://github.com/BayraktarLab/cell2location/issues/408

## Implementation Details

### Poisson
- **Knuth** algorithm for λ < 10
- **Hörmann transformed rejection** for λ ≥ 10

### Dirichlet
- Samples `Gamma(alpha, 1)` via `_standard_gamma`
- Normalizes along the last dimension

### RNG
- Uses **Philox4x32-10** for reproducibility

## Files Changed
- `aten/src/ATen/native/mps/kernels/Distributions.metal`
- `aten/src/ATen/native/mps/operations/Distributions.mm`
- `aten/src/ATen/native/native_functions.yaml`
- `test/test_mps.py`

## Test Plan

```bash
python test/test_mps.py TestMPS.test_poisson_dirichlet_standard_gamma
```

cc @malfet @kulinseth @albanD @DenisVieriu97
