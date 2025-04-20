# MPS Implementation of upsample_nearest3d_vec

This PR adds a Metal Performance Shaders (MPS) implementation of the `upsample_nearest3d_vec` operator for PyTorch on macOS. This implementation enables 3D nearest neighbor upsampling to run natively on Apple Silicon GPUs.

## Changes

- Added MPS implementation of `upsample_nearest3d_vec` in `aten/src/ATen/native/mps/operations/UpSample.mm`
- Added tests in `test/test_mps_upsample_nearest3d.py`
- Requires macOS 13.1 or newer due to Metal API requirements

## Implementation Details

The implementation uses a custom Metal compute shader to perform 3D nearest neighbor upsampling. The shader calculates the source coordinates for each output voxel and samples the nearest input voxel.

Key features:
- Supports both `scale_factor` and `size` parameters
- Handles non-contiguous tensors
- Supports empty tensors
- Supports both float32 and float16 data types

## Limitations

- Backward pass is not yet implemented
- Only supports upsampling (scale factors >= 1.0)
- Integer data types are not supported (Metal limitation)

## Testing

The implementation has been tested with various input shapes, scale factors, and data types. All tests pass on macOS 13.1 and newer.

## Performance

The MPS implementation provides significant performance improvements over the CPU implementation, especially for larger tensors.

## Future Work

- Implement backward pass
- Support downsampling (scale factors < 1.0)
- Optimize performance further

## Related Issues

This PR addresses the need for native MPS implementation of 3D upsampling operations, which was previously falling back to CPU.
