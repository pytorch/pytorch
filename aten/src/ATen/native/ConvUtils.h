#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <utility>

namespace at::native {

using conv_depthwise2d_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 2>);
DECLARE_DISPATCH(conv_depthwise2d_backward_fn, conv_depthwise2d_backward_stub)
using conv_depthwise3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(conv_depthwise3d_backward_fn, conv_depthwise3d_backward_stub)
using cudnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
DECLARE_DISPATCH(cudnn_convolution_backward_fn, cudnn_convolution_backward_stub)
using mps_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mps_convolution_backward_fn, mps_convolution_backward_stub)
using cudnn_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
DECLARE_DISPATCH(cudnn_convolution_transpose_backward_fn, cudnn_convolution_transpose_backward_stub)
using miopen_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_convolution_backward_fn, miopen_convolution_backward_stub)
using miopen_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_convolution_transpose_backward_fn, miopen_convolution_transpose_backward_stub)
using miopen_depthwise_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
DECLARE_DISPATCH(miopen_depthwise_convolution_backward_fn, miopen_depthwise_convolution_backward_stub)
using mkldnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mkldnn_convolution_backward_fn, mkldnn_convolution_backward_stub)
using mkldnn_convolution_transpose_fn = Tensor(*)(const Tensor&, const Tensor&, const std::optional<Tensor>&,
    IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
DECLARE_DISPATCH(mkldnn_convolution_transpose_fn, mkldnn_convolution_transpose_stub)
using mkldnn_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mkldnn_convolution_transpose_backward_fn, mkldnn_convolution_transpose_backward_stub)
using slow_conv_dilated2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated2d_backward_fn, slow_conv_dilated2d_backward_stub)
using slow_conv_dilated3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated3d_backward_fn, slow_conv_dilated3d_backward_stub)
using slow_conv_transpose2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose2d_backward_fn, slow_conv_transpose2d_backward_stub)
using slow_conv_transpose3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose3d_backward_fn, slow_conv_transpose3d_backward_stub)

namespace {
  bool is_cudnnv8_heuristic_mode_b() {
    static const bool is_cudnnv8_heuristic_mode_b = c10::utils::check_env("TORCH_CUDNN_USE_HEURISTIC_MODE_B") == true;
    return is_cudnnv8_heuristic_mode_b;
  }
}

inline bool cudnnv8_enabled_check_debug() {
  static bool cudnnv8_flag = c10::utils::check_env("TORCH_CUDNN_V8_API_DISABLED") != true;
  static bool cudnnv8_debug = c10::utils::check_env("TORCH_CUDNN_V8_API_DEBUG") == true;
  static uint8_t cudnnv8_debugcount = 0;
  if (cudnnv8_debug == 1 && cudnnv8_debugcount < 10) {
    TORCH_WARN("TORCH_CUDNN_V8_DEBUG ON, V8 ON: ", cudnnv8_flag, " TORCH_CUDNN_USE_HEURISTIC_MODE B: ", is_cudnnv8_heuristic_mode_b());
    cudnnv8_debugcount++;
  }
  return cudnnv8_flag == 1;
}

inline bool cudnnv8_use_heur_mode_b() {
  return is_cudnnv8_heuristic_mode_b();
}

// Keep in sync with py::enum_ in Module.cpp
enum class ConvBackend {
  CudaDepthwise2d,
  CudaDepthwise3d,
  Cudnn,
  CudnnTranspose,
  Empty,
  Miopen,
  MiopenDepthwise,
  MiopenTranspose,
  Onednn,
  OnednnTranspose,
  OnednnEmpty,
  NnpackSpatial,
  Overrideable,
  Slow2d,
  Slow3d,
  SlowDilated2d,
  SlowDilated3d,
  SlowTranspose2d,
  SlowTranspose3d,
  Winograd3x3Depthwise,
  Xnnpack2d,
  Mps,
  MpsTranspose,
};

// Overload for selecting the convolution backend from the full set of convolution inputs.
// This overload is exposed to python for testing, etc.
TORCH_API ConvBackend select_conv_backend(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation,
    bool transposed, SymIntArrayRef output_padding, c10::SymInt groups, const at::OptionalSymIntArrayRef bias_sizes_opt);

TORCH_API at::MemoryFormat _determine_backend_memory_format(const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend);

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

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  TORCH_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
           "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    TORCH_CHECK(false, ss.str());
  }
}


// NOTE [ Convolution checks ]
//
// NB: For many call sites, it is not strictly necessary to check all of
// these relationships (for example, for forward convolution, we compute
// the size of output ourselves, so we don't actually need to check
// output.  However, writing a single function that does everything
// means we get to reuse it for both forwards and all backwards
// variants, even when the set of "real" inputs varies.  The magic of
// relational computing!
//
// (There is one downside, which is that it is slightly harder to write
// error messages which are able to distinguish between real inputs
// (which the user can change) and computed inputs (which the user can
// only indirectly affect).  It would be an interesting exercise to
// come up with a general framework to handle such situations.)
inline void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize_symint(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

template <typename T>
inline std::vector<T> _conv_output_size(
    ArrayRef<T> input_size, ArrayRef<T> weight_size,
    ArrayRef<T> padding, ArrayRef<T> stride, ArrayRef<T> dilation = ArrayRef<T>()
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  bool has_dilation = !dilation.empty();
  auto dim = input_size.size();
  std::vector<T> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

inline std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation = IntArrayRef()
) {
  return _conv_output_size(input_size, weight_size, padding, stride, dilation);
}

inline std::vector<c10::SymInt> conv_output_size(
    SymIntArrayRef input_size, SymIntArrayRef weight_size,
    SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation = SymIntArrayRef()
) {
  return _conv_output_size(input_size, weight_size, padding, stride, dilation);
}

template <typename T>
std::vector<T> _conv_input_size(
    ArrayRef<T> output_size, ArrayRef<T> weight_size,
    ArrayRef<T> padding, ArrayRef<T> output_padding, ArrayRef<T> stride, ArrayRef<T> dilation, T groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<T> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (const auto d : c10::irange(2, dim)) {
    auto kernel = (weight_size[d] - 1) * dilation[d - 2] + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (padding[d - 2] * 2) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

inline std::vector<c10::SymInt> conv_input_size(
    SymIntArrayRef output_size, SymIntArrayRef weight_size,
    SymIntArrayRef padding, SymIntArrayRef output_padding, SymIntArrayRef stride, SymIntArrayRef dilation, c10::SymInt groups
) {
  return _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, std::move(groups));
}

inline std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  return _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, groups);
}

template <typename T>
std::vector<T> _conv_weight_size(
    ArrayRef<T> input_size, ArrayRef<T> output_size,
    ArrayRef<T> padding, ArrayRef<T> output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  auto dim = input_size.size();
  std::vector<T> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (const auto d : c10::irange(2, dim)) {
    auto kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + padding[d - 2] * 2 - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}

inline std::vector<c10::SymInt> conv_weight_size(
    SymIntArrayRef input_size, SymIntArrayRef output_size,
    SymIntArrayRef padding, SymIntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  return _conv_weight_size(input_size, output_size, padding, output_padding, stride, dilation, groups);
}

inline std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  return _conv_weight_size(input_size, output_size, padding, output_padding, stride, dilation, groups);
}

inline Tensor reshape_bias(int64_t dim, const Tensor& bias) {
  std::vector<int64_t> shape(dim, 1);
  shape[1] = -1;
  return bias.reshape(shape);
}

inline at::MemoryFormat cudnn_conv_suggest_memory_format(const at::Tensor& input, const at::Tensor& weight) {
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

// controls whether emptyCache will be called following cudnn conv benchmarking
TORCH_API void _cudnn_set_conv_benchmark_empty_cache(bool enable);
TORCH_API bool _cudnn_get_conv_benchmark_empty_cache();


inline bool miopen_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // disable NHWC for float64 input.
  if (!at::detail::getCUDAHooks().compiledWithMIOpen() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  bool can_use_miopen_channels_last_2d = false;
  // TODO: Remove PYTORCH_MIOPEN_SUGGEST_NHWC once ROCm officially supports NHWC in MIOpen
  // See #64427
  static std::optional<bool> PYTORCH_MIOPEN_SUGGEST_NHWC = c10::utils::check_env("PYTORCH_MIOPEN_SUGGEST_NHWC");

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  can_use_miopen_channels_last_2d = PYTORCH_MIOPEN_SUGGEST_NHWC &&  *PYTORCH_MIOPEN_SUGGEST_NHWC && (
            ( (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
            (weight_memory_format == at::MemoryFormat::ChannelsLast) )
        );

  bool can_use_miopen_channels_last_3d = false;

  return can_use_miopen_channels_last_2d || can_use_miopen_channels_last_3d;
}

inline bool onednn_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // disable NHWC for float64 input.
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  // disable NHWC for OnednnCPU tensor.
  if (input.is_onednn() || weight.is_onednn()) {
    return false;
  }

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  bool can_use_onednn_channels_last_2d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast);

  bool can_use_onednn_channels_last_3d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast3d);

  return can_use_onednn_channels_last_2d || can_use_onednn_channels_last_3d;
}

inline bool thnn_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  bool can_use_thnn_channels_last_2d = input.device().is_cpu() && (
      (input_memory_format  == at::MemoryFormat::ChannelsLast) || (
       weight_memory_format == at::MemoryFormat::ChannelsLast));

  return can_use_thnn_channels_last_2d;
}

inline bool xpu_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // check layout only for xpu tensor.
  if (!input.is_xpu() || !weight.is_xpu()) {
    return false;
  }

  // disable NHWC for float64 input.
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  bool can_use_xpu_channels_last_2d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast);

  bool can_use_xpu_channels_last_3d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast3d);

  return can_use_xpu_channels_last_2d || can_use_xpu_channels_last_3d;
}

} // namespace at::native
