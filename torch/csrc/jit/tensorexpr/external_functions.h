#pragma once

#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/Export.h>
#include <cstdint>
#include <vector>

#define FOR_ALL_EXTERNAL_FUNCTIONS(_)   \
  _(nnc_aten_adaptive_avg_pool2d)       \
  _(nnc_aten_addmm)                     \
  _(nnc_aten_conv2d)                    \
  _(nnc_aten_conv1d)                    \
  _(nnc_aten_conv1d_out)                \
  _(nnc_aten_dequantize)                \
  _(nnc_aten_dequantize_out)            \
  _(nnc_aten_embedding)                 \
  _(nnc_aten_matmul)                    \
  _(nnc_aten_mv)                        \
  _(nnc_aten_mm)                        \
  _(nnc_aten_mean)                      \
  _(nnc_aten_max_red)                   \
  _(nnc_aten_max_red_out)               \
  _(nnc_aten_quantized_conv1d)          \
  _(nnc_aten_quantized_conv1d_out)      \
  _(nnc_aten_quantized_conv2d)          \
  _(nnc_aten_quantized_conv2d_out)      \
  _(nnc_aten_quantized_conv2d_relu)     \
  _(nnc_aten_quantized_conv2d_relu_out) \
  _(nnc_aten_quantized_linear)          \
  _(nnc_aten_quantized_linear_out)      \
  _(nnc_aten_quantized_linear_relu)     \
  _(nnc_aten_quantized_add)             \
  _(nnc_aten_quantized_cat)             \
  _(nnc_aten_quantized_mul)             \
  _(nnc_aten_quantized_mul_out)         \
  _(nnc_aten_quantized_mul_scalar)      \
  _(nnc_aten_quantized_mul_scalar_out)  \
  _(nnc_aten_quantized_relu)            \
  _(nnc_aten_quantized_sigmoid)         \
  _(nnc_aten_quantized_sigmoid_out)     \
  _(nnc_aten_quantize_per_tensor)       \
  _(nnc_aten_quantize_per_tensor_out)   \
  _(nnc_aten_triangular_solve)          \
  _(nnc_aten_upsample_nearest2d)        \
  _(nnc_aten_upsample_nearest2d_out)    \
  _(nnc_prepacked_conv2d_clamp_run)     \
  _(nnc_prepacked_linear_clamp_run)

#define DECLARE_EXTERNAL_FUNCTION(NAME) \
  TORCH_API void NAME(                  \
      int64_t bufs_num,                 \
      void** buf_data,                  \
      int64_t* buf_ranks,               \
      int64_t* buf_dims,                \
      int64_t* buf_strides,             \
      int8_t* buf_dtypes,               \
      int64_t args_num,                 \
      int64_t* extra_args);

namespace torch {
namespace jit {
namespace tensorexpr {
struct QIData final {
  double scale;
  int64_t zero;
  c10::ScalarType scalarType;
};
std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =
        c10::nullopt);

std::vector<at::Tensor> constructTensors2(
    int64_t bufs_in_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =
        c10::nullopt,
    size_t bufs_out_num = 0);

#ifdef C10_MOBILE
extern "C" {
#endif
void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept;

FOR_ALL_EXTERNAL_FUNCTIONS(DECLARE_EXTERNAL_FUNCTION)
#if AT_MKLDNN_ENABLED()
DECLARE_EXTERNAL_FUNCTION(nnc_mkldnn_prepacked_conv_run);
#endif

TORCH_API void nnc_aten_free(int64_t bufs_num, void** ptrs) noexcept;

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#undef DECLARE_EXTERNAL_FUNCTION
