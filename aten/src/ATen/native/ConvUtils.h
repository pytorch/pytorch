#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

namespace at { namespace native {

using conv_depthwise2d_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 2>);
DECLARE_DISPATCH(conv_depthwise2d_backward_fn, conv_depthwise2d_backward_stub);
using conv_depthwise3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(conv_depthwise3d_backward_fn, conv_depthwise3d_backward_stub);
using cudnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
DECLARE_DISPATCH(cudnn_convolution_backward_fn, cudnn_convolution_backward_stub);
using cudnn_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
DECLARE_DISPATCH(cudnn_convolution_transpose_backward_fn, cudnn_convolution_transpose_backward_stub);
using miopen_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_convolution_backward_fn, miopen_convolution_backward_stub);
using miopen_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_convolution_transpose_backward_fn, miopen_convolution_transpose_backward_stub);
using miopen_depthwise_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_depthwise_convolution_backward_fn, miopen_depthwise_convolution_backward_stub);
using mkldnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mkldnn_convolution_backward_fn, mkldnn_convolution_backward_stub);
using slow_conv_dilated2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated2d_backward_fn, slow_conv_dilated2d_backward_stub);
using slow_conv_dilated3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated3d_backward_fn, slow_conv_dilated3d_backward_stub);
using slow_conv_transpose2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose2d_backward_fn, slow_conv_transpose2d_backward_stub);
using slow_conv_transpose3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose3d_backward_fn, slow_conv_transpose3d_backward_stub);

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool cudnn_enabled;
  bool allow_tf32;

  bool is_strided() const;
  bool is_dilated() const;
  bool is_padded() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  bool is_stride_nonpos() const;
  void view1d_as_2d();
  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight) const;
  bool needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_miopen(const at::Tensor& input, const at::Tensor& weight, bool bias_defined) const;
  bool use_mkldnn(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_nnpack(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_xnnpack(const at::Tensor& input, const at::Tensor& weight,
                   const c10::optional<IntArrayRef> bias_sizes_opt) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

enum class ConvBackend {
  CudaDepthwise2d,
  CudaDepthwise3d,
  Cudnn,
  CudnnTranspose,
  Empty,
  Miopen,
  MiopenDepthwise,
  MiopenTranspose,
  Mkldnn,
  MkldnnEmpty,
  NnpackSpatial,
  Overrideable,
  Slow2d,
  Slow3d,
  SlowDilated2d,
  SlowDilated3d,
  SlowTranspose2d,
  SlowTranspose3d,
  Winograd3x3Depthwise,
  Xnnpack2d
};

// Function to select the convolution backend based on the inputs and params.
// This overload is used within the convolution internals but not exposed to python.
// NB: The forward pass provides a bias tensor while the backward pass provides
// a bool indicating whether the bias is defined. This is done to save memory by
// avoiding saving the full bias tensor for backward.
TORCH_API ConvBackend select_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<IntArrayRef> bias_sizes_opt,
    const bool need_backward,
    const ConvParams& params);

// Overload for selecting the convolution backend from the full set of convolution inputs.
// This overload is exposed to python for testing, etc.
TORCH_API ConvBackend select_conv_backend(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups);

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

static inline std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation = IntArrayRef()
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  bool has_dilation = dilation.size() > 0;
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

static inline std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (const auto d : c10::irange(2, dim)) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

static inline std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  auto dim = input_size.size();
  std::vector<int64_t> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (const auto d : c10::irange(2, dim)) {
    int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + 2 * padding[d - 2] - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}

static inline Tensor reshape_bias(int64_t dim, const Tensor& bias) {
  std::vector<int64_t> shape(dim, 1);
  shape[1] = -1;
  return bias.reshape(shape);
}

static inline at::MemoryFormat cudnn_conv_suggest_memory_format(const at::Tensor& input, const at::Tensor& weight) {
  // disable NHWC for float64 input.
  if (!at::detail::getCUDAHooks().compiledWithCuDNN() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return at::MemoryFormat::Contiguous;
  }
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();
  auto weight_ndim = weight.ndimension();

  bool can_use_cudnn_channels_last_2d = (cudnn_version >= 7603) && (weight_ndim == 4) && (
    (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
    (weight_memory_format == at::MemoryFormat::ChannelsLast)
  );
  if (can_use_cudnn_channels_last_2d) {
    return at::MemoryFormat::ChannelsLast;
  }

  bool can_use_cudnn_channels_last_3d = (cudnn_version >= 8005) && (weight_ndim == 5) && (
    (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
    (weight_memory_format == at::MemoryFormat::ChannelsLast3d)
  );
  if (can_use_cudnn_channels_last_3d) {
    return at::MemoryFormat::ChannelsLast3d;
  }

  return at::MemoryFormat::Contiguous;
}

static inline bool miopen_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // disable NHWC for float64 input.
  if (!at::detail::getCUDAHooks().compiledWithMIOpen() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  bool can_use_miopen_channels_last_2d = false;
#if defined(USE_ROCM) && (ROCM_VERSION >= 40300)
  // TODO: Remove PYTORCH_MIOPEN_SUGGEST_NHWC once ROCm officially supports NHWC in MIOpen
  // See #64427
  static c10::optional<bool> PYTORCH_MIOPEN_SUGGEST_NHWC = c10::utils::check_env("PYTORCH_MIOPEN_SUGGEST_NHWC");

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  can_use_miopen_channels_last_2d = PYTORCH_MIOPEN_SUGGEST_NHWC &&  *PYTORCH_MIOPEN_SUGGEST_NHWC && (
            ( (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
            (weight_memory_format == at::MemoryFormat::ChannelsLast) )
        );
#endif

  bool can_use_miopen_channels_last_3d = false;

  return can_use_miopen_channels_last_2d || can_use_miopen_channels_last_3d;
}

}} // namespace at::native
