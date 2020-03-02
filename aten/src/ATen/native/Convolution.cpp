#include <limits>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cpu/DepthwiseConvKernel.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/ConvUtils.h>

#include <ATen/Config.h>
#if AT_NNPACK_ENABLED()
#include "nnpack.h"
#endif

constexpr int MIOPEN_DIM_MAX = 5;

namespace at { namespace native {

DEFINE_DISPATCH(convolution_depthwise3x3_winograd_stub);

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
  bool use_miopen(const at::Tensor& input, bool bias_defined) const;
  bool use_mkldnn(const at::Tensor& input) const;
  bool use_nnpack(const at::Tensor& input) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

std::ostream& operator<<(std::ostream & out, const ConvParams& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << IntArrayRef{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << IntArrayRef{params.output_padding}
      << "  groups = " << params.groups
      << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic
      << "  cudnn_enabled = " << params.cudnn_enabled
      << "}";
  return out;
}

auto ConvParams::is_strided() const -> bool {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

auto ConvParams::is_dilated() const -> bool {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

auto ConvParams::is_padded() const -> bool {
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

auto ConvParams::is_output_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_output_padding_big() const -> bool {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |= (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

auto ConvParams::is_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_stride_nonpos() const -> bool {
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

auto ConvParams::use_cpu_depthwise3x3_winograd(
    const at::Tensor& input, const at::Tensor& weight) const -> bool {
#ifdef __ARM_NEON__
  // Currently only 3x3 depthwise convolutions on tensors of float are supported.
  return (input.ndimension() == 4) &&
         (input.size(1) == groups) &&
         (weight.ndimension() == 4 ) &&
         (weight.size(0) % input.size(1) == 0) &&
         (weight.size(2) == 3) &&
         (weight.size(3) == 3) &&
         (input.device().type() == c10::DeviceType::CPU) &&
         (input.scalar_type() == at::kFloat) &&
         input.is_contiguous() &&
         (weight.device().type() == c10::DeviceType::CPU) &&
         (weight.scalar_type() == at::kFloat) &&
         weight.is_contiguous() &&
         !is_strided() &&
         !is_dilated() &&
         !transposed;
#else
  return false;
#endif
}

auto ConvParams::needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const -> bool {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  int64_t numel_input = input.numel();
  // empty input
  if (numel_input == 0) {
    return false;
  }
  // input size can not be reduced to the range of int by splitting the batch dim
  int64_t n = input.size(0);
  if (numel_input / n > int_max) {
    return true;
  }
  // output size can not be reduced to the range of int by splitting the batch dim
  int64_t outsize = 1;
  if (transposed) {
    std::vector<int64_t> o = conv_input_size(input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
    for (int64_t i = 1; i < o.size(); i++) {
      outsize *= o[i];
    }
  } else {
    std::vector<int64_t> o = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
    for (int64_t i = 1; i < o.size(); i++) {
      outsize *= o[i];
    }
  }
  return outsize > int_max;
}

auto ConvParams::use_cudnn(const at::Tensor& input, const at::Tensor& weight) const -> bool {
  if (needs_64bit_indexing_no_split(input, weight)) {
    return false;
  }
  if (!detail::getCUDAHooks().compiledWithCuDNN()) {
    return false;
  }
  if (!input.is_cuda() || !cudnn_enabled) {
    return false;
  }
  if (deterministic && is_dilated()) {
    // cudnn doesn't support deterministic dilated convolution fully yet
    return false;
  }
  if (is_dilated()) {
    return detail::getCUDAHooks().supportsDilatedConvolutionWithCuDNN() && !is_output_padding_big();
  }
  return !is_output_padding_big();
}

auto ConvParams::use_miopen(const at::Tensor& input, bool bias_defined) const -> bool {

  return ((input.scalar_type() == at::kFloat) || (input.scalar_type() == at::kHalf) || (input.scalar_type() == at::kBFloat16))
         && detail::getCUDAHooks().compiledWithMIOpen()
         && input.is_cuda()
         && input.dim() <= MIOPEN_DIM_MAX
         && !(groups > 1 && is_dilated()) // MIOpen currently does not support dilation with groups of size > 1
         && !(input.scalar_type() == at::kBFloat16 && bias_defined) // MIOpen currently doesn't support bias with bfloat16
         && cudnn_enabled
         ;
}

auto ConvParams::use_mkldnn(const at::Tensor& input) const -> bool {
#if AT_MKLDNN_ENABLED()
  if (!at::globalContext().userEnabledMkldnn()) {
    return false;
  }
  return (input.is_mkldnn()) || // input is mkldnn Tensor
    (input.options().backend() == at::Backend::CPU &&
     input.scalar_type() == kFloat && // only on CPU Float Tensors
     !is_dilated() && // doesn't support dilation
     !transposed && // or transposed tensors
     input.ndimension() == 4); // must be in NCHW format
#endif
  return false;
}
auto ConvParams::use_nnpack(const at::Tensor& input) const -> bool {
#if AT_NNPACK_ENABLED()
  return at::_nnpack_available() &&
         input.options().backend() == at::Backend::CPU &&
         input.scalar_type() == kFloat && // only on CPU Float Tensors
         !is_dilated() && // or dilation
         !transposed &&   // or transposed tensors
         input.ndimension() == 4 // must be in NCHW format
#if !defined(C10_MOBILE) && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)
         && input.size(0) >= 16 // ensure large enough batch size to ensure perf, tuneable
#endif
     ;
#endif
  return false;
}

// We currently only have depthwise support for the case where groups ==
// nInputPlane and nInputPlane == nOutputPlane (the latter due to the lack of
// a depthwise multiplier)
auto ConvParams::is_depthwise(
        const at::Tensor& input, const at::Tensor& weight) const -> bool {
  return input.is_cuda() &&
         !transposed &&
         input.ndimension() == 4 &&
         input.size(1) == groups &&
         groups > 1 && // no point if there is only a single group
         weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels
}

// Check workload to activate fast depthwise FP16 cudnn conv kernels
bool check_cudnn_depthwise_workload(const at::Tensor& input, int stride) {
  int w = input.size(3);  // same as h
  int ch = input.size(1);
  int bs = input.size(0);
  if (stride==1) {
    if (w >= 7) {
      // All batch sizes and nb_channels
      if (w >= 112) {
        return true;
      }

      // large nb_channels
      if (ch >= 1024) {
        if (w >= 56) {
          return true;
        } else if (bs >= 32) {
          return true;
        }
      }

      // batch_size specific
      if (bs >= 128) {
        if (ch >= 512) {
          return true;
        } else if (ch >= 64) {
          if (w >= 14) {
            return true;
          }
        } else if ((ch >= 32) && (w >=28)) {
          return true;
        }
      } else if (bs >= 64) {
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 32) && (w >= 28)) {
          return true;
        }
      } else if (bs >= 32) {
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 128) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 16) {
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        }
        if ((ch >= 256) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 8) {
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if ((ch >= 64) && (w >= 56)) {
          return true;
        }
      }
    }
  } else if (stride==2) {
    if (ch < 256) {
      return false;
    }

    if (w >= 7) {
      if (bs >= 128) {
        if (ch >= 1024) {
          return true;
        } else if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 64) {
        if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 32) {
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 16) {
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 8) {
        if ((ch >= 1024) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 1) {
        if ((ch >= 512) && (w >=112)) {
          return true;
        }
      }
    }
  }
  return false;
}
// Use cudnn for FP16 depthwise convolutions
auto ConvParams::use_cudnn_depthwise(
        const at::Tensor& input, const at::Tensor& weight) const -> bool {
  if (detail::getCUDAHooks().supportsDepthwiseConvolutionWithCuDNN()) {
    long cudnn_version = detail::getCUDAHooks().versionCuDNN();
    bool kernel_cond =  (cudnn_version >= 7600 &&
                         use_cudnn(input, weight) &&
                         input.scalar_type() == kHalf && // only for FP16
                         weight.scalar_type() == kHalf &&
                         is_depthwise(input, weight) &&
                         weight.size(2) == weight.size(3) && // only square kernels
                         input.size(2) >= 7 && // min width/height 7
                         !is_dilated() && // no dilation supported
                         stride[0] == stride[1] && // equal strides
                         ((weight.size(3) == 3) || (weight.size(3) == 1)) &&
                         input.size(1) >= 32); // min 32 channels supported)
    if (kernel_cond) {
      return check_cudnn_depthwise_workload(input, stride[0]);
    } else {
      return false;
    }
  } else {
    return false;
  }
}

static void check_shape_forward(const at::Tensor& input,
                                const c10::IntArrayRef& weight_sizes, const at::Tensor& bias,
                                const ConvParams& params, bool input_is_mkldnn) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  int64_t groups = params.groups;
  auto padding = params.padding;
  auto output_padding = params.output_padding;
  auto stride = params.stride;
  auto dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
           input.sizes(), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size [", weight_sizes,
           "] instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
             "Given groups=", groups, ", weight of size ", weight_sizes,
             ", expected input", input.sizes(), " to have ",
             (weight_sizes[1] * groups), " channels, but got ", input.size(1),
             " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
             "Given weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
             ", but got bias of size ", bias.sizes(), " instead");

    for (int i = 2; i < k; ++i) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected input", input.sizes(), " to have ", weight_sizes[0],
             " channels, but got ", input.size(1), " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
             ", but got bias of size ", bias.sizes(), " instead");
  }
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4,
           "expected 4D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.squeeze(2);
}


static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}


at::Tensor conv1d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         false, {0}, groups);
}

at::Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         false, {{0, 0}}, groups);
}

at::Tensor conv3d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         false, {{0, 0, 0}}, groups);
}

at::Tensor conv_transpose1d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         true, output_padding, groups);
}

at::Tensor conv_transpose2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         true, output_padding, groups);
}

at::Tensor conv_transpose3d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         true, output_padding, groups);
}

at::Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  auto& ctx = at::globalContext();
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups,
                          ctx.benchmarkCuDNN(), ctx.deterministicCuDNN(), ctx.userEnabledCuDNN());
}

at::Tensor convolution_overrideable(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  AT_ERROR("You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use torch::RegisterOperators() to override this function ");
}

at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled) {

  const bool input_is_mkldnn = input_r.is_mkldnn();
  auto input = input_r;
  auto weight = weight_r;
  auto bias = bias_r;
  auto k = weight.ndimension();
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  std::vector<int64_t> weight_sizes_mkl;
  c10::IntArrayRef weight_sizes = weight.sizes();
  if (input_is_mkldnn && (k == input.ndimension() + 1)) {
    k = input.ndimension();
    weight_sizes_mkl.resize(k);
    weight_sizes_mkl[0] = weight.size(0) * weight.size(1);
    std::copy_n(
        weight.sizes().cbegin() + 2, k - 1, weight_sizes_mkl.begin() + 1);
    weight_sizes = c10::IntArrayRef(weight_sizes_mkl);
  }
  int64_t dim = k - 2;
  

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;
  params.cudnn_enabled = cudnn_enabled;

  check_shape_forward(input, weight_sizes, bias, params, input_is_mkldnn);

  if (input.size(0) == 0) {    
    // don't send empty inputs through backends
    // but need to compute correct output size first and set up history for params
    std::vector<int64_t> o;
    if (!params.transposed) {
      o = conv_output_size(input.sizes(), weight_sizes, params.padding,
                           params.stride, params.dilation);
    } else {
      o = conv_input_size(input.sizes(), weight_sizes, params.padding,
                          params.output_padding, params.stride, params.dilation,
                          params.groups);
    }
    auto weight_view = at::_unsafe_view(weight, -1);
    auto out = input*weight_view[0];
    if (bias.defined())
      out = out + bias[0];
    return out.view(o);
  }

  if (k == 3) {
    // avoid accidentally going through NHWC for permuted 3d input.
    if (!input_is_mkldnn) {
      input = input.contiguous();
    }
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  at::MemoryFormat cudnn_memory_format = cudnn_conv_use_channels_last(input, weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  Tensor output;
  if (params.is_depthwise(input, weight)) {
      /* output.resize_(output_size(input, weight)); */

      auto kernel_size = weight.sizes().slice(2);
      auto stride = params.stride;
      auto padding = params.padding;
      auto dilation = params.dilation;
      if (params.use_cudnn_depthwise(input, weight)) {
        output = at::cudnn_convolution(
            input.contiguous(cudnn_memory_format), weight,
            padding, stride, dilation, params.groups, params.benchmark, params.deterministic);
        if (bias.defined()) {
          output = output + reshape_bias(input.dim(), bias);
        }

      } else if (params.use_miopen(input, bias.defined())){
        output = at::miopen_depthwise_convolution(
            input.contiguous(), weight, bias,
            padding, stride, dilation, params.groups, params.benchmark, params.deterministic);
      } else {
          output = at::thnn_conv_depthwise2d(input.contiguous(), weight, kernel_size, bias, stride, padding, dilation);
      }
  } else if (params.use_cudnn(input, weight)) {
    TORCH_CHECK(input.options().type_equal(weight.options()),
             "Input type (", input.toString(), ") and weight type (", weight.toString(),
             ") should be the same");
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
             "Input type (", input.toString(), ") and bias type (", bias.toString(),
             ") should be the same");

    if (params.transposed) {
      output = at::cudnn_convolution_transpose(
          input.contiguous(cudnn_memory_format), weight,
          params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
      if (bias.defined()) {
        output = output + reshape_bias(input.dim(), bias);
      }
    } else {
      output = at::cudnn_convolution(
          input.contiguous(cudnn_memory_format), weight,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
      if (bias.defined()) {
        output = output + reshape_bias(input.dim(), bias);
      }
    }
  } else if (params.use_miopen(input, bias.defined())) {
    TORCH_CHECK(input.options().type_equal(weight.options()),
             "Input type (", input.toString(), ") and weight type (", weight.toString(),
             ") should be the same");
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
             "Input type (", input.toString(), ") and bias type (", bias.toString(),
             ") should be the same");

    if (params.transposed) {
      output = at::miopen_convolution_transpose(
          input.contiguous(), weight, bias,
          params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    } else {
      output = at::miopen_convolution(
          input.contiguous(), weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    }
  } else if (params.use_mkldnn(input)) {
#if AT_MKLDNN_ENABLED()
    TORCH_CHECK(input.options().type_equal(weight.options()),
             "Input type (", input.toString(), ") and weight type (", weight.toString(),
             ") should be the same");
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
             "Input type (", input.toString(), ") and bias type (", bias.toString(),
             ") should be the same");
    if (!input_is_mkldnn) {
      output = at::mkldnn_convolution(input.contiguous(), weight.contiguous(), bias.defined() ? bias.contiguous() : bias,
                                      params.padding, params.stride, params.dilation, params.groups);
    } else {
      // do not call contiguous on mkldnn tensor
      output = at::mkldnn_convolution(input, weight, bias,
                                      params.padding, params.stride, params.dilation, params.groups);
    }
#endif
  } else if (input.device().type() == c10::DeviceType::CPU || input.device().type() == c10::DeviceType::CUDA) {
    if (params.use_cpu_depthwise3x3_winograd(input, weight)) {
      output = convolution_depthwise3x3_winograd_stub(
        input.device().type(), input, weight, bias, params.stride, params.padding, params.groups);
    } else if (params.groups == 1) {
      output = at::_convolution_nogroup(
          input.contiguous(), weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
    } else {
      std::vector<Tensor> outputs(params.groups);
      input = input.contiguous();
      for (int g = 0; g < params.groups; ++g) {
        auto input_g = subtensor(input, 1, params.groups, g);
        auto weight_g = subtensor(weight, 0, params.groups, g);
        auto bias_g = subtensor(bias, 0, params.groups, g);
        outputs[g] = at::_convolution_nogroup(
            input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
      }
      output = at::cat(outputs, 1);
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    output = at::convolution_overrideable(input, weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups);
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

// A generic function for convolution implementations which don't
// natively implement groups (e.g., not CuDNN).
at::Tensor _convolution_nogroup(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding) {

  ConvParams params;
  params.stride = stride.vec();
  params.padding = padding.vec();
  params.dilation = dilation.vec();
  params.transposed = transposed;
  params.output_padding = output_padding.vec();
  params.groups = 1;
  params.benchmark = false;
  params.deterministic = false;
  params.cudnn_enabled = false;

  auto dim = input.ndimension();
  auto dilated = params.is_dilated();
  auto kernel_size = weight.sizes().slice(2);

  if (params.transposed) {
    if (dim == 4) {
      return at::slow_conv_transpose2d(
          input, weight, kernel_size, bias,
          stride, padding, output_padding, dilation);
    } else if (dim == 5) {
      return at::slow_conv_transpose3d(
        input, weight, kernel_size, bias,
        stride, padding, output_padding, dilation);
      }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        return at::slow_conv_dilated2d(
            input, weight, kernel_size, bias,
            stride, padding, dilation);
      } else {  /* dim == 4, non-dilated */
        if (params.use_nnpack(input)) {
#if AT_NNPACK_ENABLED()
          return at::_nnpack_spatial_convolution(
              input, weight, bias, padding, stride);
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          return at::thnn_conv2d(
              input, weight, kernel_size, bias,
              stride, padding);
        }
      }
    } else if (dim == 5 && (input.is_cuda() || dilated)) {
      return at::slow_conv_dilated3d(
          input, weight, kernel_size, bias,
          stride, padding, dilation);
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels
         for non-dilated case here */
      return at::slow_conv3d(
          input, weight, kernel_size, bias,
          stride, padding);
    }
  }

  AT_ERROR("unsupported ConvNd parameters");
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
  AT_ERROR("You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use torch::RegisterOperators() to override this function ");
  return std::tuple<Tensor, Tensor, Tensor>(
          at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
          at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
          at::empty({}));
}

static Tensor subvariable(const Tensor& var, int dim, int groups, int g) {
  int64_t n = var.sizes()[dim] / groups;
  auto result = var.narrow(dim, n * g, n);
  return result;
}

std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(
    const Tensor& ggI, const Tensor& ggW_r, const Tensor& ggb,
    const Tensor& gO_r, const Tensor& weight_r, const Tensor& input,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled,
    std::array<bool, 3> output_mask) {

  auto ggW = ggW_r;
  auto gO = gO_r;
  auto weight = weight_r;

  ConvParams params;
  params.stride = stride_.vec();
  params.padding = padding_.vec();
  params.dilation = dilation_.vec();
  params.transposed = transposed_;
  params.output_padding = output_padding_.vec();
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;
  params.cudnn_enabled = cudnn_enabled;

  // Compute ggO = conv(ggI, w) + conv(i, ggW) + ggb
  Tensor ggO;
  if (input.numel() != 0) {
    if (ggI.defined()) {
      if (weight.is_cuda()) {
        weight = weight.contiguous();
      }
      ggO = at::_convolution(ggI, weight, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups, params.benchmark, params.deterministic, params.cudnn_enabled);
    }

    if (ggW.defined()) {
      if (ggW.is_cuda()) {
        ggW = ggW.contiguous();
      }
      auto ggW_term = at::_convolution(input, ggW, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups, params.benchmark, params.deterministic, params.cudnn_enabled);
      if (ggO.defined()) {
        ggO = ggO + ggW_term;
      } else {
        ggO = ggW_term;
      }
    }
  } else {
    ggO = at::zeros_like(gO, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (ggb.defined()) {
    // View as (1, ggb.size(0), 1, 1...)

    // Expand
    std::vector<int64_t> new_size(gO.ndimension(), 1);
    new_size[1] = ggb.sizes()[0];
    auto ggb_contiguous = ggb.contiguous();
    auto ggb_view = ggb_contiguous.view(new_size);

    // Expand
    auto ggb_expanded = ggb_view.expand(gO.sizes());

    if (ggO.defined()) {
      ggO = ggO + ggb_expanded;
    } else {
      ggO = ggb_expanded;
    }
  }

  // Compute gW = conv(ggI, gO)
  Tensor gW;
  if (ggI.defined()) {

    // Modified params with correct padding
    ConvParams gw_conv_params(params);

    // Disable groups as they are handled separately
    auto groups = gw_conv_params.groups;
    gw_conv_params.groups = 1;
    std::swap(gw_conv_params.dilation, gw_conv_params.stride);

    // Transpose gO and ggI to accumulate over batch
    auto gOt = gO.transpose(0, 1);
    auto ggIt = ggI.transpose(0, 1);

    Tensor gWt;
    // Compute conv
    if (input.numel() != 0) {
      if (groups == 1) {

        if (gOt.is_cuda()) {
          gOt = gOt.contiguous();
        }

        // Compute conv
        if (params.transposed) {
          gw_conv_params.transposed = false;
          gWt = at::_convolution(gOt, ggIt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled);
        } else {
          gWt = at::_convolution(ggIt, gOt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled);
        }
      } else {
        std::vector<Tensor> gWt_list(groups);
        for (int g = 0; g < groups; ++g) {
          auto ggIt_g = subvariable(ggIt, 0, groups, g);
          auto gOt_g = subvariable(gOt, 0, groups, g);
          if (gOt_g.is_cuda()) {
            gOt_g = gOt_g.contiguous();
          }

          // Compute conv
          if (params.transposed) {
            gw_conv_params.transposed = false;
            gWt_list[g] = at::_convolution(gOt_g, ggIt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled);
          } else {
            gWt_list[g] = at::_convolution(ggIt_g, gOt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled);
          }
        }

        gWt = at::cat(gWt_list, 1);
      }

      // Transpose gW to match chan_in and chan_out
      gW = gWt.transpose(0, 1);

      // narrow gW to only relevant portion
      // we do it this way instead of narrowing the input itself because
      // the ConvForward kernels don't support asymmetric padding.
      auto gW_size = gW.sizes();
      auto w_size = weight.sizes();
      for (size_t i = 2; i < gW_size.size(); ++i) {
        if (gW_size[i] > w_size[i]) {
            gW = gW.narrow(i, 0, w_size[i]);
            gW_size = gW.sizes();
        }
      }
    }
  } else {
    gW = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  // Compute gI = convT(ggW, gO.t()) if !transposed
  //         gI = conv(go, ggw)      if transposed
  Tensor gI;
  if (input.numel() != 0) {
    if (ggW.defined()) {
      ConvParams gi_conv_params(params);
      gi_conv_params.transposed = !params.transposed;

      if (params.transposed) {
        if (gO.is_cuda()) {
          gO = gO.contiguous();
        }
        gI = at::_convolution(gO, ggW, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups, gi_conv_params.benchmark, gi_conv_params.deterministic, gi_conv_params.cudnn_enabled);

        // narrow gI to only relevant portion
        // we do it this way because negative output_padding is not supported
        // TODO: figure out if we can narrow gO and save some compute,
        // rather than narrowing the computed gI
        auto gI_size = gI.sizes();
        auto i_size = input.sizes();
        for (size_t i = 2; i < gI_size.size(); ++i) {
          if (gI_size[i] > i_size[i]) {
            gI = gI.narrow(i, 0, i_size[i]);
            gI_size = gI.sizes();
          }
        }
      } else {
        auto groups = gi_conv_params.groups;
        gi_conv_params.groups = 1;
        // swap stride and dilation
        std::swap(gi_conv_params.dilation, gi_conv_params.stride);

        auto ggWt = ggW.transpose(0, 1);
        auto gOt = gO.transpose(0, 1);

        // calculate output_padding
        // TODO: figure out why this needs to be computed...
        auto kernel_size = weight.sizes().slice(2);
        auto input_shape = input.sizes().slice(2);
        auto grad_output_shape = gO.sizes().slice(2);

        if (kernel_size.size() == 1) {
          auto expected_input_shape = (kernel_size[0] - 1) * gi_conv_params.stride[1]
            - 2 * gi_conv_params.padding[1]
            + (gi_conv_params.dilation[1] * (grad_output_shape[0] - 1) + 1);
          if (expected_input_shape != input_shape[0]) {
            gi_conv_params.output_padding[1] = input_shape[0] - expected_input_shape;
          }
        } else {
          for(size_t i = 0; i < kernel_size.size(); ++i) {
            // Check if whole input has been used or not
            auto expected_input_shape = (kernel_size[i] - 1) * gi_conv_params.stride[i]
              - 2 * gi_conv_params.padding[i]
              + (gi_conv_params.dilation[i] * (grad_output_shape[i] - 1) + 1);
            if (expected_input_shape != input_shape[i]) {
              gi_conv_params.output_padding[i] = input_shape[i] - expected_input_shape;
            }
          }
        }

        Tensor gIt;
        if (params.groups == 1) {
          if (gOt.is_cuda()) {
            gOt = gOt.contiguous();
          }

          gIt = at::_convolution(ggWt, gOt, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups, gi_conv_params.benchmark, gi_conv_params.deterministic, gi_conv_params.cudnn_enabled);
        } else {
          std::vector<Tensor> gIt_list(params.groups);
          for (int g = 0; g < groups; ++g) {
            auto ggWt_g = subvariable(ggWt, 1, groups, g);
            auto gOt_g = subvariable(gOt, 0, groups, g);
            if (gOt_g.is_cuda()) {
              gOt_g = gOt_g.contiguous();
            }

            gIt_list[g] = at::_convolution(ggWt_g, gOt_g, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups, gi_conv_params.benchmark, gi_conv_params.deterministic, gi_conv_params.cudnn_enabled);
          }

          gIt = at::cat(gIt_list, 0);
        }

        gI = gIt.transpose(0, 1);
      }
    }
  } else {
    gI = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (output_mask[0] && !ggO.defined()) ggO = at::zeros_like(gO, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (output_mask[1] && !gI.defined()) gI = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (output_mask[2] && !gW.defined()) gW = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  return std::tuple<Tensor,Tensor,Tensor>{ggO, gI, gW};
}

}} // at::native
