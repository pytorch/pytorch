# MPS CTC requested changes

Repo: `/Users/nhossain/Downloads/Projects/pytorch-mps-ctc-loss-minimal`
Source branch: `aspinto/pytorch:mps-ctc-loss`

## Scope cleanup applied
Removed unrelated changes from the branch (net diff vs base now excludes these):
- `.vscode/settings_recommended.json`
- `aten/src/ATen/CMakeLists.txt` whitespace-only change
- `aten/src/ATen/templates/TensorBody.h` (`Tensor::mps()` addition)
- `torch/csrc/api/include/torch/python.h` (`Module::mps()` addition)
- ad-hoc scripts under `test/aspinto/*`

## Kept PR scope (only CTC MPS op)
- `aten/src/ATen/native/mps/LossCTC.mm`
- `aten/src/ATen/native/mps/kernels/LossCTC.metal`
- `aten/src/ATen/native/native_functions.yaml`

## malfet-requested technical fixes applied
1. Reworked Metal encoder argument binding to use `mtl_setArgs` instead of repetitive `setBuffer`/`setBytes` blocks.
2. Added half/bfloat16 support path in wrappers by promoting compute to float32 and casting outputs back.
3. Tightened target dtype handling: accept only int32/int64 targets; normalize to int64 for MPS kernel path.
4. Fixed an additional kernel bug: `input_length` in `ctc_loss_zero_padded_gradients` now uses `int64_t` (not `scalar_t`).
5. Corrected argument ordering/indexing risks by contiguous `mtl_setArgs` calls (including nonblank collect and zero-padded gradients paths).

## Tests added (PyTorch style)
- `test/test_mps.py`
  - `test_ctc_loss_matches_cpu_float32`
  - `test_ctc_loss_low_precision_matches_cpu`
  - Validates forward/backward parity (CPU reference vs MPS), including tensor/list length inputs, unbatched case, and zero-target-length coverage.
- `test/test_nn.py`
  - Removed stale MPS expected-failure markers for CTC-related tests now that MPS CTC exists.
  - Updated `test_CTCLoss_empty_target` to use MPS-supported dtype on MPS.
  - Kept gradcheck-only CTC path skipped on MPS (double-only gradcheck path).

## Current branch scope
Net diff against `pytorch/pytorch:main`:
- `A aten/src/ATen/native/mps/LossCTC.mm`
- `A aten/src/ATen/native/mps/kernels/LossCTC.metal`
- `M aten/src/ATen/native/native_functions.yaml` (minimal CTC dispatch additions)
- `M test/test_nn.py`
- `M test/test_mps.py`
- `A requested changes.md`
