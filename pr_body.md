## Summary

This PR adds MPS backend support for the `channel_shuffle` operation.

Channel shuffle rearranges channels from groups for inter-group communication in shuffle-based networks (e.g., ShuffleNet, ShuffleNetV2). This is commonly used in efficient CNN architectures for mobile devices.

## Changes

The `channel_shuffle` function is device-agnostic - it calls `native_channel_shuffle` which uses `CompositeImplicitAutograd` (`math_channel_shuffle`) with basic operations (`view`, `permute`, `contiguous`, `reshape`) that are all MPS-compatible.

By adding MPS to the dispatch, we enable direct `channel_shuffle` calls on MPS tensors.

## Test Plan

Added `test_channel_shuffle` in `test_mps.py` testing:
- Multiple dtypes (float32, float16)
- Various group sizes (1, 2, 3, 4)
- 3D, 4D, and 5D tensors
- Comparison with CPU results

cc @kulinseth @albanD @malfet @DenisVieriu97 @razarmehr
