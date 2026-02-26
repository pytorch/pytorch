(mps_environment_variables)=
# MPS Environment Variables

**PyTorch Environment Variables**


| Variable                         | Description |
|----------------------------------|-------------|
| `PYTORCH_DEBUG_MPS_ALLOCATOR`   | If set to `1`, set allocator logging level to verbose. |
| `PYTORCH_MPS_LOG_PROFILE_INFO`  | Set log options bitmask to `MPSProfiler`. See `LogOptions` enum in `aten/src/ATen/mps/MPSProfiler.h`. |
| `PYTORCH_MPS_TRACE_SIGNPOSTS`   | Set profile and signpost bitmasks to `MPSProfiler`. See `ProfileOptions` and `SignpostTypes`. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | High watermark ratio for MPS allocator. Default is 1.7. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | Low watermark ratio for MPS allocator. Default is 1.4 (unified) or 1.0 (discrete). |
| `PYTORCH_MPS_FAST_MATH`         | If `1`, enables fast math for MPS kernels. See section 1.6.3 in the [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). |
| `PYTORCH_MPS_PREFER_METAL`      | If `1`, uses metal kernels instead of MPS Graph APIs. Used for matmul. |
| `PYTORCH_ENABLE_MPS_FALLBACK`   | If `1`, falls back to CPU when MPS ops aren't supported. |

```{note}
**high watermark ratio** is a hard limit for the total allowed allocations

- `0.0` : disables high watermark limit (may cause system failure if system-wide OOM occurs)
- `1.0` : recommended maximum allocation size (i.e., device.recommendedMaxWorkingSetSize)
- `>1.0`: allows limits beyond the device.recommendedMaxWorkingSetSize

e.g., value 0.95 means we allocate up to 95% of recommended maximum
allocation size; beyond that, the allocations would fail with OOM error.

**low watermark ratio** is a soft limit to attempt limiting memory allocations up to the lower watermark
level by garbage collection or committing command buffers more frequently (a.k.a, adaptive commit).
Value between 0 to m_high_watermark_ratio (setting 0.0 disables adaptive commit and garbage collection)
e.g., value 0.9 means we 'attempt' to limit allocations up to 90% of recommended maximum
allocation size.
```
