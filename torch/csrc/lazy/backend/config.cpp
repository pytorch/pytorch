#include <torch/csrc/lazy/core/config.h>

// TODO(whc) either deprecate this, or use it for all shape inference
C10_DEFINE_int(
    torch_lazy_tensors_shape_cache_size,
    4096,
    "Set the size for the shape cache used for shape inference");

// TODO(whc) unclear if this is useful, has only been tested as true
C10_DEFINE_bool(
    torch_lazy_tensors_tensor_update_sync,
    true,
    "Use synchronous copy inside _copy_from op");

// TODO(whc) we need to hook up these flags in a more useful way
// possibly also keep LTC_TS_CUDA env working?
C10_DEFINE_bool(
    torch_lazy_tensors_cuda,
    false,
    "Use cuda device for torchscript backend (instead of CPU)");
