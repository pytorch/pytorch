#pragma once
#include <ATen/Parallel.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Unfold2d.h>
#include <ATen/native/Unfold3d.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

namespace at { namespace native {

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
  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) const;
  bool needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_miopen(const at::Tensor& input, const at::Tensor& weight, bool bias_defined) const;
  bool use_mkldnn(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_nnpack(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_xnnpack(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) const;
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
TORCH_API ConvBackend select_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
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

// Computes columns used in slow 2D kernel computation.
// This is computed separately in the forward and backward passes.
// Note that the input tensor is assumed to be a 4D tensor of shape (N, C, H, W).
static Tensor compute_columns2d(
    const Tensor& input,
    const ConvParams& params,
    IntArrayRef kernel_size) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = params.padding[0];
  const int64_t pad_width = params.padding[1];
  const int64_t stride_height = params.stride[0];
  const int64_t stride_width = params.stride[1];
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t batch_size = input.size(0);

  Tensor columns;
  if ((input.ndimension() == 4) && (kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
      (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
    // Columns are just a view on the input for this special case.
    columns = input.view(
      {batch_size,
      n_input_plane,
      output_height * output_width}).detach();
  } else if (input.is_cuda()) {
    // CUDA kernel computes column data internally so just allocate a correctly-shaped tensor.
    columns = at::empty({n_input_plane * kernel_width * kernel_height, output_height * output_width}, input.options());
  } else {
    columns = at::im2col(input, kernel_size, params.dilation, params.padding, params.stride);
  }

  return columns;
}

// Computes columns used in slow 3D kernel computation.
// This is computed separately in the forward and backward passes.
// Note that the input tensor is assumed to be a 5D tensor of shape (N, C, D, H, W).
static Tensor compute_columns3d(
    const Tensor& input,
    const ConvParams& params,
    IntArrayRef kernel_size,
    const int64_t groups,
    const int64_t n_output_plane) {
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = params.padding[0];
  const int64_t pad_height = params.padding[1];
  const int64_t pad_width = params.padding[2];
  const int64_t stride_depth = params.stride[0];
  const int64_t stride_height = params.stride[1];
  const int64_t stride_width = params.stride[2];
  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t batch_size = input.size(0);

  Tensor columns;
  if ((kernel_depth == 1) && (kernel_height == 1) && (kernel_width == 1) &&
      (pad_depth == 0) && (pad_height == 0) && (pad_width == 0) &&
      (stride_depth == 1) && (stride_height == 1) && (stride_width == 1) && (groups == 1)) {
    // Columns are just a view on the input for this special case.
    columns = input.view({batch_size, n_input_plane, output_height * output_width * output_depth}).detach();
  } else if (input.is_cuda()) {
    // CUDA kernel computes column data internally so just allocate a correctly-shaped tensor.
    columns = at::empty({n_output_plane * kernel_width * kernel_height * kernel_depth,
                        input_depth * input_height * input_width}, input.options());
  } else {
    columns = at::empty({batch_size,
                        n_input_plane * kernel_depth * kernel_height * kernel_width,
                        output_depth * output_height * output_width},
                        input.options());

    const Tensor input_ = input.contiguous();
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, input_.scalar_type(), "compute_columns3d", [&] {
      auto input_a = input_.accessor<scalar_t, 5>();
      auto columns_a = columns.accessor<scalar_t, 3>();

      constexpr int64_t CONV3D_GRAIN_SALT = 20;

      at::parallel_for(0, batch_size, CONV3D_GRAIN_SALT, [&](int64_t start, int64_t end) {
        for (int64_t t = start; t < end; t++) {
          auto input_t = input_a[t];
          auto columns_t = columns_a[t];
          Unfold3dCopyCPU(
            c10::CppTypeToScalarType<scalar_t>::value,
            input_t.data(),
            n_input_plane,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            pad_depth,
            pad_height,
            pad_width,
            columns_t.data());
          }
      });
    });
  }

  return columns;
}

}} // namespace at::native
