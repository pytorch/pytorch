#include <torch/csrc/lazy/core/config.h>

// TODO(whc) either deprecate this, or use it for all shape inference
C10_DEFINE_int(
    torch_lazy_ts_shape_cache_size,
    4096,
    "Set the size for the shape cache used for shape inference");
