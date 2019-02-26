#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/Config.h>

#if AT_NNPACK_ENABLED()
#include "nnpack.h"
#endif

static const int MIOPEN_DIM_MAX = 4;

namespace at { namespace native {

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
  void view1d_as_2d();
  bool use_cudnn(const at::Tensor& input) const;
  bool use_miopen(const at::Tensor& input) const;
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


auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

auto ConvParams::use_cudnn(const at::Tensor& input) const -> bool {
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

auto ConvParams::use_miopen(const at::Tensor& input) const -> bool {

  return ((input.type().scalarType() == at::kFloat) || (input.type().scalarType() == at::kHalf))
         && detail::getCUDAHooks().compiledWithMIOpen()
         && input.is_cuda()
         && input.dim() <= MIOPEN_DIM_MAX
         && !(groups > 1 && is_dilated()) // MIOpen currently does not support dilation with groups of size > 1
         && !transposed
         && (dilation.at(0) == dilation.at(1)) //MIOpen currently does not support assymetric dilation values.
         && (stride.at(0) == stride.at(1)) //Line 549 & 635 (swapping stride and dilation values) leads to assymetric dilation values.
         ;
}

auto ConvParams::use_mkldnn(const at::Tensor& input) const -> bool {
#if AT_MKLDNN_ENABLED()
  return input.type().backend() == at::Backend::CPU &&
         input.type().scalarType() == kFloat && // only on CPU Float Tensors
         !is_dilated() && // doesn't support dilation
         !transposed && // or transposed tensors
         input.ndimension() == 4; // must be in NCHW format
#endif
  return false;
}
auto ConvParams::use_nnpack(const at::Tensor& input) const -> bool {
#if AT_NNPACK_ENABLED()
  return at::_nnpack_available() &&
         input.type().backend() == at::Backend::CPU &&
         input.type().scalarType() == kFloat && // only on CPU Float Tensors
         !is_strided() && // doesn't support strides
         !is_dilated() && // or dilation
         !transposed &&   // or transposed tensors
         input.ndimension() == 4 // must be in NCHW format
#if !C10_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)
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

static void check_input_shape_forward(const at::Tensor& input,
                                      const at::Tensor& weight, const at::Tensor& bias,
                                      int64_t groups, bool transposed) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight.ndimension();

  AT_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight.sizes(), ", but got ", k, "-dimensional input of size ",
           input.sizes(), " instead");
  AT_CHECK(weight.size(0) >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight.sizes(), " instead");
  AT_CHECK(weight.size(0) % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size ", weight.sizes(),
           " instead");

  if (!transposed) {
    AT_CHECK(input.size(1) == (weight.size(1) * groups),
             "Given groups=", groups, ", weight of size ", weight.sizes(),
             ", expected input", input.sizes(), " to have ",
             (weight.size(1) * groups), " channels, but got ", input.size(1),
             " channels instead");
    AT_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight.size(0)),
             "Given weight of size ", weight.sizes(),
             ", expected bias to be 1-dimensional with ", weight.size(0), " elements",
             ", but got bias of size ", bias.sizes(), " instead");
  } else { // transposed
    AT_CHECK(input.size(1) == weight.size(0),
             "Given transposed=", transposed, ", weight of size ", weight.sizes(),
             ", expected input", input.sizes(), " to have ", weight.size(0),
             " channels, but got ", input.size(1), " channels instead");
    AT_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight.size(1) * groups),
             "Given transposed=", transposed, ", weight of size ", weight.sizes(),
             ", expected bias to be 1-dimensional with ", weight.size(1) * groups, " elements",
             ", but got bias of size ", bias.sizes(), " instead");
  }
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  AT_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  AT_CHECK(tensor.ndimension() == 4,
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

static inline std::vector<int64_t> convolution_expand_param_if_needed(
  IntArrayRef list_param, const char *param_name, int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t) list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

std::tuple<Tensor, Tensor, Tensor, int64_t> _convolution_impl_index(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled) {
  auto input = input_r.contiguous();
  auto weight = weight_r;
  auto bias = bias_r;
  auto k = weight.ndimension();
  int64_t dim = k - 2;
  int64_t impl_index = 0;

  AT_CHECK(dim > 0, "weight should have at least three dimensions");

  Tensor save1 = at::zeros({}, input.type());
  Tensor save2 = at::zeros({}, input.type());

  ConvParams params;
  params.stride = convolution_expand_param_if_needed(stride_, "stride", dim);
  params.padding = convolution_expand_param_if_needed(padding_, "padding", dim);
  params.dilation = convolution_expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = convolution_expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;
  params.cudnn_enabled = cudnn_enabled;

  AT_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  AT_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");

  check_input_shape_forward(input, weight, bias, params.groups, params.transposed);

  if (k == 3) {
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  auto output = at::empty({0}, input.options());

  if (params.is_depthwise(input, weight)) {
      /* output.resize_(output_size(input, weight)); */
      impl_index = 0;
      auto kernel_size = weight.sizes().slice(2);
      auto stride = params.stride;
      auto padding = params.padding;
      auto dilation = params.dilation;

      output = at::thnn_conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation);
  } else if (params.use_cudnn(input)) {
    AT_CHECK(input.type() == weight.type(),
             "Input type (", input.type().toString(), ") and weight type (", weight.type().toString(),
             ") should be the same");
    AT_CHECK(!bias.defined() || (input.type() == bias.type()),
             "Input type (", input.type().toString(), ") and bias type (", bias.type().toString(),
             ") should be the same");

    if (params.transposed) {
      impl_index = 1;
      output = at::cudnn_convolution_transpose(
          input, weight, bias,
          params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    } else {
      impl_index = 2;
      output = at::cudnn_convolution(
          input, weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    }
  } else if (params.use_miopen(input)) {
    AT_CHECK(input.type() == weight.type(),
             "Input type (", input.type().toString(), ") and weight type (", weight.type().toString(),
             ") should be the same");
    AT_CHECK(!bias.defined() || (input.type() == bias.type()),
             "Input type (", input.type().toString(), ") and bias type (", bias.type().toString(),
             ") should be the same");

    if (params.transposed) {
      impl_index = 3;
      output = at::miopen_convolution_transpose(
          input, weight, bias,
          params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    } else {
      impl_index = 4;
      output = at::miopen_convolution(
          input, weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    }
  } else if (params.use_mkldnn(input)) {
#if AT_MKLDNN_ENABLED()
    impl_index = 5;
    AT_CHECK(input.type() == weight.type(),
             "Input type (", input.type().toString(), ") and weight type (", weight.type().toString(),
             ") should be the same");
    AT_CHECK(!bias.defined() || (input.type() == bias.type()),
             "Input type (", input.type().toString(), ") and bias type (", bias.type().toString(),
             ") should be the same");
    output = at::mkldnn_convolution(input, weight.contiguous(), bias.defined() ? bias.contiguous() : bias,
                                    params.padding, params.stride, params.dilation, params.groups);
#endif
  } else {
    if (params.groups == 1) {
      // impl_index 6 - 12 are in _convolution_nogroup
      auto returned_tuple = at::_convolution_nogroup(
          input, weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
      output = std::get<0>(returned_tuple);
      save1 = std::get<1>(returned_tuple);
      save2 = std::get<2>(returned_tuple);
      impl_index = std::get<3>(returned_tuple);
    } else {
      std::vector<Tensor> outputs(params.groups);
      std::vector<Tensor> save1s(params.groups);
      std::vector<Tensor> save2s(params.groups);
      for (int g = 0; g < params.groups; ++g) {
        auto input_g = subtensor(input, 1, params.groups, g);
        auto weight_g = subtensor(weight, 0, params.groups, g);
        auto bias_g = subtensor(bias, 0, params.groups, g);
        auto returned_tuple = at::_convolution_nogroup(
            input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
        outputs[g] = std::get<0>(returned_tuple);
        save1s[g] = std::get<1>(returned_tuple);
        save2s[g] = std::get<2>(returned_tuple);
        impl_index = std::get<3>(returned_tuple);
      }
      output = at::cat(outputs, 1);
      save1 = at::cat(save1s, 1);
      save2 = at::cat(save2s, 1);
    }
  }

  if (k == 3) {
    output = view3d(output);
  }

  return std::make_tuple(output, save1, save2, impl_index);
}

// A generic function for convolution implementations which don't
// natively implement groups (e.g., not CuDNN).
std::tuple<Tensor, Tensor, Tensor, int64_t> _convolution_nogroup(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding) {

  Tensor save1 = at::zeros({}, input.type());
  Tensor save2 = at::zeros({}, input.type());

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

  int64_t impl_index = 6;
  if (params.transposed) {
    if (dim == 4) {
      return std::tuple_cat(at::thnn_conv_transpose2d_forward(
          input, weight, kernel_size, bias,
          stride, padding, output_padding, dilation), std::make_tuple(6));
    } else if (dim == 5) {
      return std::tuple_cat(at::thnn_conv_transpose3d_forward(
        input, weight, kernel_size, bias,
        stride, padding, output_padding, dilation), std::make_tuple(7));
      }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        return std::tuple_cat(at::thnn_conv_dilated2d_forward(
            input, weight, kernel_size, bias,
            stride, padding, dilation), std::make_tuple(8));
      } else {  /* dim == 4, non-dilated */
        if (params.use_nnpack(input)) {
#if AT_NNPACK_ENABLED()
          return std::make_tuple(at::_nnpack_spatial_convolution(
              input, weight, bias, padding), save1, save2, 9);
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          return std::tuple_cat(at::thnn_conv2d_forward(
              input, weight, kernel_size, bias,
              stride, padding), std::make_tuple(10));
        }
      }
    } else if (dim == 5 && (input.is_cuda() || dilated)) {
      return std::tuple_cat(at::thnn_conv_dilated3d_forward(
          input, weight, kernel_size, bias,
          stride, padding, dilation), std::make_tuple(11));
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels
         for non-dilated case here */
      return std::tuple_cat(at::thnn_conv3d_forward(
          input, weight, kernel_size, bias,
          stride, padding), std::make_tuple(12));
    }
  }

  AT_ERROR("unsupported ConvNd parameters");
}

at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled) {
  return std::get<0>(at::_convolution_impl_index(input_r, weight_r, bias_r, stride_,
              padding_, dilation_, transposed_, output_padding_, groups_, benchmark,
              deterministic, cudnn_enabled));
}

std::tuple<Tensor, Tensor, Tensor> _convolution_nogroup_impl_index_backward(
    int64_t impl_index, const Tensor& self, const Tensor& grad,
    const Tensor& weight, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding,
    IntArrayRef output_padding, IntArrayRef dilation,
    std::array<bool, 3> output_mask, const Tensor& save1, const Tensor& save2) {
  Tensor grad_input, grad_weight, grad_bias;
  if (impl_index == 6) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv_transpose2d_backward(
        grad, self, weight, kernel_size, stride, padding,
        output_padding, dilation, /*columns=*/save1, /*ones=*/save2,
        output_mask);
  } else if (impl_index == 7) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv_transpose3d_backward(
        grad, self, weight, kernel_size, stride, padding,
        output_padding, dilation, /*finput=*/save1, /*fgrad_input=*/save2,
        output_mask);
  } else if (impl_index == 8) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv_dilated2d_backward(
        grad, self, weight, kernel_size, stride, padding,
        dilation, /*columns=*/save1, /*ones=*/save2,
        output_mask);
  } else if (impl_index == 9) {
    grad_input = at::_nnpack_spatial_convolution_backward_input(
        self, grad, weight, padding);
    grad_weight = at::_nnpack_spatial_convolution_backward_weight(
        self, weight.sizes(), grad, padding);
  } else if (impl_index == 10) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv2d_backward(
        grad, self, weight, kernel_size, stride, padding,
        /*finput=*/save1, /*fgrad_input=*/save2,
        output_mask);
  } else if (impl_index == 11) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv_dilated3d_backward(
        grad, self, weight, kernel_size, stride, padding,
        dilation, /*columns=*/save1, /*ones=*/save2,
        output_mask);
  } else if (impl_index == 12) {
    std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv3d_backward(
        grad, self, weight, kernel_size, stride, padding,
        /*finput=*/save1, /*fgrad_input=*/save2,
        output_mask);
  } else {
    // NEVER REACH HERE
    AT_ERROR("unsupported backward impl index");
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> _convolution_impl_index_backward(
    int64_t impl_index, const Tensor& self, const Tensor& grad_r,
    const Tensor& weight_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    IntArrayRef output_padding_, bool transposed_, int64_t groups_,
    bool benchmark, bool deterministic, std::array<bool, 3> output_mask,
    const Tensor& save1, const Tensor& save2) {

  auto grad = grad_r;
  auto input = self.contiguous();
  auto weight = weight_r;
  auto k = weight_r.ndimension();
  int64_t dim = k - 2;

  ConvParams params;
  params.stride = convolution_expand_param_if_needed(stride_, "stride", dim);
  params.padding = convolution_expand_param_if_needed(padding_, "padding", dim);
  params.dilation = convolution_expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = convolution_expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;

  if (k == 3) {
    params.view1d_as_2d();
    grad = view4d(grad);
    input = view4d(input);
    weight = view4d(weight);
  }

  Tensor grad_input, grad_weight, grad_bias;

  if (impl_index == 0) {
    auto kernel_size = weight.sizes().slice(2);
    auto stride = params.stride;
    auto padding = params.padding;
    auto dilation = params.dilation;
    std::array<bool, 2> out_mask{ output_mask.at(0), output_mask.at(1)};
    std::tie(grad_input, grad_weight) = at::thnn_conv_depthwise2d_backward(
        grad.contiguous(), input, weight, kernel_size, stride, padding, dilation,
        out_mask);
    grad_bias = grad.contiguous().view({grad.size(0), grad.size(1), -1}).sum(0).sum(1);
  } else if (impl_index == 1) {
    std::tie(grad_input, grad_weight, grad_bias) = at::cudnn_convolution_transpose_backward(
        input, grad, weight, params.padding, params.output_padding, params.stride, params.dilation,
        params.groups, params.benchmark, params.deterministic, output_mask);
  } else if (impl_index == 2) {
    std::tie(grad_input, grad_weight, grad_bias) = at::cudnn_convolution_backward(
        input, grad, weight, params.padding, params.stride, params.dilation, params.groups,
        params.benchmark, params.deterministic, output_mask);
  } else if (impl_index == 3) {
    std::tie(grad_input, grad_weight, grad_bias) = at::miopen_convolution_transpose_backward(
        input, grad, weight, params.padding, params.output_padding, params.stride, params.dilation,
        params.groups, params.benchmark, params.deterministic, output_mask);
  } else if (impl_index == 4) {
    std::tie(grad_input, grad_weight, grad_bias) = at::miopen_convolution_backward(
        input, grad, weight, params.padding, params.stride, params.dilation, params.groups,
        params.benchmark, params.deterministic, output_mask);
  } else if (impl_index == 5) {
    std::tie(grad_input, grad_weight, grad_bias) = at::mkldnn_convolution_backward(
        input, grad, weight, params.padding, params.stride, params.dilation, params.groups,
        output_mask);
  } else {
    auto kernel_size = weight.sizes().slice(2);
    if (params.groups == 1) {
        std::tie(grad_input, grad_weight, grad_bias) = _convolution_nogroup_impl_index_backward(
            impl_index, input, grad, weight, kernel_size, params.stride, params.padding, params.output_padding,
            params.dilation, output_mask, save1, save2);
    } else {
    // multi groups
      std::vector<Tensor> grad_inputs(params.groups);
      std::vector<Tensor> grad_weights(params.groups);
      std::vector<Tensor> grad_biases(params.groups);
      for (int g = 0; g < params.groups; ++g) {
        auto grad_g = subtensor(grad, 1, params.groups, g);
        auto weight_g = subtensor(weight, 0, params.groups, g);
        auto grads = _convolution_nogroup_impl_index_backward(
            impl_index, input, grad_g, weight_g, kernel_size, params.stride, params.padding, params.output_padding,
            params.dilation, output_mask, save1, save2);
        grad_inputs[g] = std::get<0>(grads);
        grad_weights[g] = std::get<1>(grads);
        grad_biases[g] = std::get<2>(grads);
      }

      grad_input = at::cat(grad_inputs, 1);
      grad_weight = at::cat(grad_weights, 0);
      grad_bias = at::cat(grad_biases, 0);
    }
  }

  if (k == 3) {
    grad_input = view3d(grad_input);
    grad_weight = view3d(grad_weight);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
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

  // Compute gI = convT(ggW, gO.t()) if !transposed
  //         gI = conv(go, ggw)      if transposed
  Tensor gI;
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

  if (output_mask[0] && !ggO.defined()) ggO = at::zeros_like(gO);
  if (output_mask[1] && !gI.defined()) gI = at::zeros_like(input);
  if (output_mask[2] && !gW.defined()) gW = at::zeros_like(weight);

  return std::tuple<Tensor,Tensor,Tensor>{ggO, gI, gW};
}

}} // at::native
