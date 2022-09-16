#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_backward_stub);

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

// follow check rules from native/Convolution.cpp without transpose supported
static void check_shape_forward(const Tensor& input,
                                const Tensor& weight,
                                const Tensor& bias,
                                const IntArrayRef& padding,
                                const IntArrayRef& stride,
                                const IntArrayRef& dilation,
                                const int64_t groups) {
#define MKLDNN_CONV_ARG_CHECK(IT, OP) std::any_of(IT.begin(), IT.end(), [](auto x) { return x OP 0; })
  auto is_padding_neg = MKLDNN_CONV_ARG_CHECK(padding, <);
  auto is_stride_nonpos = MKLDNN_CONV_ARG_CHECK(stride, <=);
  auto is_dilation_nonpos = MKLDNN_CONV_ARG_CHECK(dilation, <=);
#undef MKLDNN_CONV_ARG_CHECK
  TORCH_CHECK(!is_padding_neg, "negative padding is not supported");
  TORCH_CHECK(!is_stride_nonpos, "non-positive stride is not supported");
  TORCH_CHECK(!is_dilation_nonpos, "non-positive dilation is not supported");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");

  int64_t k = input.ndimension();
  const IntArrayRef& weight_sizes = weight.sizes();
  int64_t weight_dim = weight_sizes.size();

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
  TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
              "Given groups=", groups, ", weight of size ", weight_sizes,
              ", expected input", input.sizes(), " to have ",
              (weight_sizes[1] * groups), " channels, but got ", input.size(1),
              " channels instead");
  TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
              "Given weight of size ", weight_sizes,
              ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
              ", but got bias of size ", bias.sizes(), " instead");

  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  bool kernel_size_correct = true;

  for (const auto i : c10::irange(2, k)) {
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
    std::string separator = "";

    for (int i = 0, len = input_shape.size(); i < len; ++i) {
      input_ss << separator << input_shape[i];
      kernel_ss << separator << kernel_shape[i];
      separator = " x ";
    }

    TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). "
                "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
  }
}

#define MKLDNNTensor(itensor, options)                                  \
  new_with_itensor_mkldnn(                                              \
      std::move(itensor),                                               \
      optTypeMetaToScalarType(options.dtype_opt()),                     \
      options.device_opt())

// Note [MKLDNN Convolution Memory Formats]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MKLDNN has 3 types of memory formats in convolution:
//
// In case memory format passed from PyTorch (aka. user layout)
// differs from the internal layout which MKLDNN used, a `reorder` is needed;
// otherwise when user layout is identical to internal layout,
// MKLDNN uses a memory `view` upon an existing CPU tensor.
//
// 1. NCHW (CPU tensor, contiguous)
//  input reorder:  NCHW(user) -> Blocked(internal)
//  weight reorder: OIHW(user) -> Blocked(internal)
//  output reorder: Blocked(internal) -> NCHW(user)
//
// 2. NHWC: (CPU tensor, channels last)
//  input view:     NHWC(user) -> NHWC(internal)
//  weight reorder: OHWI(user) -> Blocked(internal)
//  output view:    NHWC(internal) -> NHWC(user)
//
// 3. Blocked (MKLDNN tensor):
//  By explicitly converting a tensor to mkldnn, e.g. `x.to_mkldnn()`,
//  blocked format will propagate between layers. Input, output will be in blocked format.
//
//  For inference case, weight can be prepacked into blocked format by
//  (so as to save weight reoder overhead):
//      model = torch.utils.mkldnn.to_mkldnn(model)
//
//  For training case, grad_output can be CPU tensor or MKLDNN tensor,
//  but weight/bias and grad_weight/grad_bias are always CPU tensor.
//

static inline at::MemoryFormat mkldnn_convolution_memory_format(int64_t dims, bool is_channels_last) {
   auto memory_format =  at::MemoryFormat::Contiguous;
   if (is_channels_last) {
      memory_format = dims == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
   }
   return memory_format;
}

Tensor mkldnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  if (input_t.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_convolution: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  check_shape_forward(input_t, weight_t, bias, padding, stride, dilation, groups);

  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);

  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  auto output_sizes = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
  auto output = at::empty({0}, input.options());

  const ideep::tensor x = itensor_from_tensor(input);
  const ideep::tensor w = itensor_from_tensor(weight);

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::convolution_forward::compute_v3(
        x,
        w,
        b,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        is_channels_last);
  } else {
    ideep::convolution_forward::compute_v3(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        is_channels_last);
  }

  if (input.is_mkldnn()) {
    return MKLDNNTensor(y, input.options());
  } else if (!is_channels_last) {
    return mkldnn_to_dense(MKLDNNTensor(y, input.options()));
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  auto grad_input = at::empty({0}, grad_output.options());

  auto grad_y = itensor_from_tensor(grad_output);
  auto w = itensor_view_from_dense(weight);

  ideep::tensor grad_x;
  if (is_channels_last) {
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_from_tensor(grad_input);
  }
  ideep::convolution_backward_data::compute_v2(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups,
      is_channels_last);

  if (grad_output.is_mkldnn()) {
    return MKLDNNTensor(grad_x, grad_output.options());
  } else if (!is_channels_last){
    return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
  } else {
    TORCH_INTERNAL_ASSERT(grad_x.get_desc().is_nhwc());
    return grad_input;
  }
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  const ideep::tensor grad_y = itensor_from_tensor(grad_output);
  const ideep::tensor x = itensor_from_tensor(input);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  } else {
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  }

  if (!is_channels_last) {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  } else {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(at::MemoryFormat::ChannelsLast),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);

  Tensor input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  Tensor weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2], is_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2], is_channels_last);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_backward_stub, &mkldnn_convolution_backward);

}}  // namespace at::native

#endif
