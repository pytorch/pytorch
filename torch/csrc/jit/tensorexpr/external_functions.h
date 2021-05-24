#pragma once

#include <ATen/Functions.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cstdint>
#include <vector>

#define FOR_ALL_EXTERNAL_FUNCTIONS(_) \
  _(nnc_aten_conv2d)                  \
  _(nnc_aten_matmul)                  \
  _(nnc_aten_mv)                      \
  _(nnc_aten_mm)                      \
  _(nnc_aten_adaptive_avg_pool2d)     \
  _(nnc_aten_mean)                    \
  _(nnc_aten_addmm)

#define DECLARE_EXTERNAL_FUNCTION(NAME) \
  TORCH_API void NAME(                  \
      int64_t bufs_num,                 \
      void** buf_data,                  \
      int64_t* buf_ranks,               \
      int64_t* buf_dims,                \
      int8_t* buf_dtypes,               \
      int64_t args_num,                 \
      int64_t* extra_args);

namespace torch {
namespace jit {
namespace tensorexpr {

std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes);

#ifdef C10_MOBILE
extern "C" {
#endif

FOR_ALL_EXTERNAL_FUNCTIONS(DECLARE_EXTERNAL_FUNCTION)

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#undef DECLARE_EXTERNAL_FUNCTION
