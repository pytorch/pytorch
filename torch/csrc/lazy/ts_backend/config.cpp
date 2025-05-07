#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/ts_backend/config.h>

// TODO(whc) unclear if this is useful, has only been tested as true
C10_DEFINE_bool(
    torch_lazy_ts_tensor_update_sync,
    true,
    "Use synchronous copy inside _copy_from op")

// TODO(whc) we need to hook up these flags in a more useful way
// possibly also keep LTC_TS_CUDA env working?
C10_DEFINE_bool(
    torch_lazy_ts_cuda,
    false,
    "Use cuda device for torchscript backend (instead of CPU)")
