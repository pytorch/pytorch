#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#else
#include <ATen/ops/_add_relu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mkldnn_convolution_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_backward_stub);
REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_transpose_stub);
REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_transpose_backward_stub);

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <c10/util/irange.h>

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

static bool mkldnn_conv_enabled_fpmath_mode_bf16(){
  return at::globalContext().float32Precision("mkldnn", "conv") == "bf16" &&
      mkldnn_bf16_device_check();
}


static inline at::MemoryFormat mkldnn_convolution_memory_format(int64_t dims, bool is_channels_last) {
   auto memory_format =  at::MemoryFormat::Contiguous;
   if (is_channels_last) {
      memory_format = dims == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
   }
   return memory_format;
}

static void _mkldnn_convolution_out(
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& bias,
    std::vector<int64_t>& output_sizes,
    ideep::tensor& y,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef padding,
    int64_t groups,
    bool is_channels_last,
    const ideep::attr_t& op_attr) {
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);
  const ideep::tensor w = itensor_from_tensor(weight, /*from_const_data_ptr*/true);
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias, /*from_const_data_ptr*/true);
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
        is_channels_last,
        op_attr);
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
        is_channels_last,
        op_attr);
  }
}

static Tensor _mkldnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool use_channels_last,
    c10::string_view attr = "none",
    torch::List<std::optional<at::Scalar>> scalars =
        torch::List<std::optional<at::Scalar>>(),
    std::optional<c10::string_view> algorithm = std::nullopt) {
  ideep::attr_t op_attr = ideep::attr_t();
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    TORCH_CHECK(
        it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
    op_attr = it->second(scalars, algorithm);
  }
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  mkldnn_check_low_precision(input_t.scalar_type(), "mkldnn_convolution");

  int64_t dim = input_t.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);

  check_shape_forward(input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups);

  auto memory_format =
      mkldnn_convolution_memory_format(input_t.ndimension(), use_channels_last);

  auto output_sizes = conv_output_size(input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
  auto output = at::empty({0}, input_t.options());
  ideep::tensor y;
  if (use_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }
  if (mkldnn_conv_enabled_fpmath_mode_bf16() &&
      input_t.scalar_type() == at::kFloat) {
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
  }
  _mkldnn_convolution_out(
      input_t,
      weight_t,
      bias,
      output_sizes,
      y,
      stride_expanded,
      dilation_expanded,
      padding_expanded,
      groups,
      use_channels_last,
      op_attr);

  if (input_t.is_mkldnn()) {
    return MKLDNNTensor(y, input_t.options());
  } else if (!use_channels_last) {
    return mkldnn_to_dense(MKLDNNTensor(y, input_t.options()));
  } else {
    return output;
  }
}

Tensor mkldnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  bool use_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  return _mkldnn_convolution(
      input_t,
      weight_t,
      bias_opt,
      padding,
      stride,
      dilation,
      groups,
      use_channels_last);
}

namespace{
Tensor mkldnn_convolution_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<c10::string_view> algorithm) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  bool use_channels_last =
      weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);
  return _mkldnn_convolution(
      input_t,
      weight_t,
      bias_opt,
      padding,
      stride,
      dilation,
      groups,
      use_channels_last,
      attr,
      scalars,
      algorithm);
}

// Fuse convolution+binary_op+unary_op for good performance, which doing such
// operation: output=unary_op(binary_op(conv(input_t, ...), other_t, alpha)).
// The binary_attr means which binary_op is, it can be "add", or
// other binary operation. the unary_attr means which unary_op is,
// it can be "relu" or other unary operation, if it is none, meaning that
// there doesn't have a unary post op. unary_scalars and unary_algorithm
// are the parameters of the unary op, such as "hardtanh" has scalar parameters,
// "gelu" has algorithm parameters.
Tensor mkldnn_convolution_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<c10::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<c10::string_view> unary_algorithm) {
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_pointwise_binary: currently only support 2d and 3d")
  TORCH_CHECK(
      !alpha.has_value() || alpha.value().to<float>() == 1.0,
      "mkldnn_convolution_pointwise_binary: the alpha value should be none or 1.0");

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float, bfloat16 or half.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  int64_t dim = input_t.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  check_shape_forward(
      input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups);

  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
  // TODO: support broadcast binary fusion.
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Binary Fusion's inputs should have same shape");
  // Only calling fusion path for channels_last path.
  // TODO: OneDNN doesn't optimize well for groups > 1 case, it will be enabled
  // at next OneDNN release.
  bool use_channels_last =
      weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);
  bool can_be_fused = groups == 1 && use_channels_last;

  c10::string_view unary_attr_value = "none";
  ideep::algorithm unary_alg;
  if (unary_attr.has_value()) {
    auto it_unary = fusion_unary_alg_map().find(unary_attr.value());
    // Now, we only support conv+binary+relu.
    TORCH_CHECK(
        it_unary != fusion_unary_alg_map().end(),
        "Unary Fusion behavior undefined.");
    unary_attr_value = unary_attr.value();
    unary_alg = it_unary->second;
  }
  auto it_binary = fusion_binary_alg_map().find(binary_attr);
  TORCH_CHECK(
      it_binary != fusion_binary_alg_map().end(),
      "Binary Fusion behavior undefined.");
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  if (can_be_fused) {
    auto memory_format =
        mkldnn_convolution_memory_format(input_t.ndimension(), true);
    auto input = input_t.contiguous(memory_format);
    auto weight =
        weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
    auto other = other_t.contiguous(memory_format);
    auto output = at::empty_like(other);
    const ideep::tensor x = itensor_from_tensor(input);
    const ideep::tensor w = itensor_from_tensor(weight);
    const ideep::tensor z = itensor_from_tensor(other);
    ideep::tensor y = itensor_from_tensor(output);
    auto output_size = other.sizes().vec();
    ideep::tag format_tag = ideep::tag::nhwc;
    if (input_t.ndimension() == 5) {
      format_tag = ideep::tag::ndhwc;
    }
    auto other_desc = ideep::tensor::desc(
        output_size, get_mkldnn_dtype(weight.scalar_type()), format_tag);

    ideep::attr_t op_attr;
    ideep::post_ops po;
    po.append_binary(it_binary->second, other_desc);
    if (unary_attr_value != "none") {
      po.append_eltwise(unary_alg, 0.f, 0.f);
    }
    op_attr.set_post_ops(po);

    if (mkldnn_conv_enabled_fpmath_mode_bf16() && input_t.scalar_type() ==at::kFloat){
      op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
    }

    if (bias.defined()) {
      const ideep::tensor b = itensor_from_tensor(bias);
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          b,
          output_size,
          y,
          stride_expanded,
          dilation_expanded,
          padding_expanded,
          padding_expanded,
          groups,
          /* is_channels_last */ true,
          op_attr);
    } else {
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          output_size,
          y,
          stride_expanded,
          dilation_expanded,
          padding_expanded,
          padding_expanded,
          groups,
          /* is_channels_last */ true,
          op_attr);
    }
    return output;
  } else {
    // Fallback case, if inputs are not channels last or have different dtype,
    // OneDNN fusion may have performance regression.
    Tensor output;
    if (weight_t.is_mkldnn()) {
      output = _mkldnn_convolution(
          input_t, weight_t, bias, padding_expanded, stride_expanded, dilation, groups, true);
    } else {
      output = at::convolution(
          input_t, weight_t, bias, stride_expanded, padding_expanded, dilation_expanded, false, 0, groups);
    }
    if (binary_attr == "add" && unary_attr_value != "none") {
      output = at::native::add_relu_(output, other_t);
      return output;
    }
    if (binary_attr == "add") {
      output.add_(other_t);
    } else if (binary_attr == "sub") {
      output.sub_(other_t);
    } else if (binary_attr == "mul") {
      output.mul_(other_t);
    } else {
      output.div_(other_t);
    }
    if (unary_attr_value != "none") {
      output.relu_();
    }
    return output;
  }
}

// Fuse convolution+binary_op+unary_op for good performance, which doing
// such operation: other_t=unary_op(binary_op(conv(input_t, ...), other_t,
// alpha)). The binary_attr means which binary_op is, it can be "add", or other
// binary operation. the unary_attr means which unary_op is, it can be "relu" or
// other unary operation, if it is none, meaning that there doesn't have a unary
// post op. unary_scalars and unary_algorithm are the parameters of the unary
// op, such as "hardtanh" has scalar parameters "gelu" has algorithm parameters.

Tensor& mkldnn_convolution_pointwise_binary_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<c10::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<c10::string_view> unary_algorithm) {
  // other_t += convolution(...), other_t = unary(other_t)
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_add_: currently only support 2d and 3d")
  TORCH_CHECK(
      binary_attr == "add",
      "mkldnn_convolution_pointwise_binary_: only support binary op fusion")
  TORCH_CHECK(
      !alpha.has_value() || alpha.value().to<float>() == 1.0,
      "mkldnn_convolution_pointwise_binary: the alpha value for the binary op should be none(meaning 1.0) or 1.0");
  TORCH_CHECK(
      !unary_attr.has_value() || unary_attr.value() == "relu",
      "mkldnn_convolution_pointwise_binary: only support none or relu unary op fusion after binary op");

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float, bfloat16 or half.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);
  int64_t dim = input_t.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  check_shape_forward(
      input_t, weight_t, bias, padding, stride, dilation, groups);

  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Add Fusion's inputs should have same shape");
  // Only calling fusion path for channels_last path and the output is contiguous tensor(channels_last).
  bool can_be_fused = (weight_t.is_mkldnn() ||
                       mkldnn_conv_use_channels_last(input_t, weight_t)) &&
      (other_t.is_contiguous(at::MemoryFormat::ChannelsLast) ||
       other_t.is_contiguous(at::MemoryFormat::ChannelsLast3d));
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  if (can_be_fused) {
    ideep::tensor y = itensor_from_tensor(other_t);
    ideep::attr_t op_attr;
    if (unary_attr.has_value()) {
      op_attr = ideep::attr_t::residual();
    } else {
      op_attr = ideep::attr_t::fuse_sum();
    }
    if (mkldnn_conv_enabled_fpmath_mode_bf16() &&
        input_t.scalar_type() == at::kFloat) {
      op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
    }
    _mkldnn_convolution_out(
        input_t,
        weight_t,
        bias,
        output_sizes,
        y,
        stride_expanded,
        dilation_expanded,
        padding_expanded,
        groups,
        true,
        op_attr);
  } else {
    // Fallback case, if inputs are not channels last or have different dtype,
    // OneDNN fusion may have performance regression.
    Tensor output;
    if (weight_t.is_mkldnn()) {
      output = _mkldnn_convolution(
          input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups, true);
    } else {
      output = at::convolution(
          input_t, weight_t, bias, stride_expanded, padding_expanded, dilation_expanded, false, 0, groups);
    }
    if (unary_attr.has_value()) {
      other_t = at::native::add_relu_(other_t, output);
    } else {
      other_t.add_(output);
    }
  }
  return other_t;
}

std::vector<int64_t> _original_deconv_weight_size(
    const Tensor& weight_t,
    int64_t groups) {
  TORCH_CHECK(weight_t.is_mkldnn() || weight_t.is_meta(), "expects weight_t to be mkldnn or meta tensor");
  // The size of weight_t is the prepacked size.
  //  Groups > 1: [g*o, i/g, ...]
  //  Groups == 1: [o, i, ...]
  // Returns original weight size in [i, o, ...]
  auto dim = weight_t.sizes().size();
  TORCH_CHECK(dim > 2);

  std::vector<int64_t> weight_IOHW_sizes(dim);
  if (groups > 1) {
    weight_IOHW_sizes[0] = weight_t.sizes()[1] * groups;
    weight_IOHW_sizes[1] = weight_t.sizes()[0] / groups;
  } else {
    weight_IOHW_sizes[0] = weight_t.sizes()[1];
    weight_IOHW_sizes[1] = weight_t.sizes()[0];
  }
  for (const auto d : c10::irange(2, dim)) {
    weight_IOHW_sizes[d] = weight_t.sizes()[d];
  }
  return weight_IOHW_sizes;
}


Tensor _mkldnn_convolution_transpose(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool use_channels_last,
    c10::string_view attr = "none",
    torch::List<std::optional<at::Scalar>> scalars =
        torch::List<std::optional<at::Scalar>>(),
    std::optional<c10::string_view> algorithm = std::nullopt) {
  ideep::attr_t op_attr = ideep::attr_t();
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    TORCH_CHECK(it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
    op_attr = it->second(scalars, algorithm);
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  mkldnn_check_low_precision(input_t.scalar_type(), "mkldnn_convolution_transpose");

  std::vector<int64_t> weight_IOHW_sizes = weight_t.is_mkldnn() ? _original_deconv_weight_size(weight_t, groups) : weight_t.sizes().vec();

  auto memory_format =
      mkldnn_convolution_memory_format(input_t.ndimension(), use_channels_last);

  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);

  int64_t dim = input.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);
  auto output_sizes = conv_input_size(input.sizes(), weight_IOHW_sizes, padding_expanded, output_padding_expanded, stride_expanded, dilation_expanded, groups);
  auto output = at::empty({0}, input.options());

  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);

  ideep::tensor w = itensor_from_tensor(weight, /*from_const_data_ptr*/true);
  if (!weight.is_mkldnn()) {
    // mkldnn transposed convolution has weight in logical order of OIHW or OIDHW,
    // while PyTorch has IOHW or IODHW, `._tranpose()` switches strides (no memory copy).
    w.transpose_(0, 1);
  }

  ideep::tensor y;
  if (use_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }

  if (mkldnn_conv_enabled_fpmath_mode_bf16() && input_t.scalar_type() ==at::kFloat){
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
  }

  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias, /*from_const_data_ptr*/true);
    ideep::convolution_transpose_forward::compute_v3(
        x,
        w,
        b,
        output_sizes,
        y,
        stride_expanded,
        padding_expanded,
        padding_r(padding_expanded, output_padding_expanded),
        dilation.vec(),
        groups,
        use_channels_last,
        op_attr);
  } else {
    ideep::convolution_transpose_forward::compute_v3(
        x,
        w,
        output_sizes,
        y,
        stride_expanded,
        padding_expanded,
        padding_r(padding_expanded, output_padding_expanded),
        dilation.vec(),
        groups,
        use_channels_last,
        op_attr);
  }
  if (input.is_mkldnn()) {
    return MKLDNNTensor(y, input.options());
  } else if (!use_channels_last) {
    return mkldnn_to_dense(MKLDNNTensor(y, input.options()));
  } else {
    return output;
  }
}

Tensor mkldnn_convolution_transpose_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<c10::string_view> algorithm) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  bool use_channels_last =
      weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);
  return _mkldnn_convolution_transpose(
      input_t,
      weight_t,
      bias_opt,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      use_channels_last,
      attr,
      scalars,
      algorithm
  );
}

Tensor mkldnn_convolution_transpose_pointwise_meta(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<c10::string_view> algorithm) {

  std::vector<int64_t> weight_IOHW_sizes = _original_deconv_weight_size(weight_t, groups);
  int64_t dim = input_t.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);
  auto output_sizes = conv_input_size(input_t.sizes(), weight_IOHW_sizes, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups);

  auto output = at::empty(output_sizes, input_t.options());
  return output;
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

  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  auto w = itensor_view_from_dense(weight, /*from_const_data_ptr*/true);

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
  const ideep::tensor grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);

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
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(memory_format),
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
  int64_t dim = input.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding_expanded, stride_expanded, dilation_expanded, groups, output_mask[2], is_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding_expanded, stride_expanded, dilation_expanded, groups, output_mask[2], is_channels_last);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}
}

REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_backward_stub, &mkldnn_convolution_backward);

namespace{
Tensor mkldnn_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups)
{
  bool use_channels_last = mkldnn_conv_use_channels_last(input, weight);
  return _mkldnn_convolution_transpose(
      input,
      weight,
      bias_opt,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      use_channels_last
  );
}

Tensor mkldnn_convolution_transpose_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  auto grad_input = at::empty({0}, grad_output.options());

  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  auto w = itensor_view_from_dense(weight, /*from_const_data_ptr*/true).transpose_(0, 1);

  ideep::tensor grad_x;
  if (is_channels_last) {
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_from_tensor(grad_input);
  }
  ideep::convolution_transpose_backward_data::compute_v3(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      padding.vec(),
      padding_r(padding, output_padding),
      dilation.vec(),
      groups,
      is_channels_last);

  if (grad_output.is_mkldnn()) {
    return MKLDNNTensor(grad_x, grad_output.options());
  } else if (!is_channels_last){
    return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
  } else {
    return grad_input;
  }
}

std::tuple<Tensor,Tensor> mkldnn_convolution_transpose_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  auto x = itensor_from_tensor(input, /*from_const_data_ptr*/true);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    ideep::convolution_transpose_backward_weights::compute_v3(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        is_channels_last);
  } else {
    ideep::convolution_transpose_backward_weights::compute_v3(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        is_channels_last);
  }

  if (!is_channels_last) {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  } else {
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(memory_format),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_transpose_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    std::array<bool,3> output_mask)
{
  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  int64_t dim = input.ndimension() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = mkldnn_convolution_transpose_backward_input(
        input.sizes(), grad_output, weight, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups, output_mask[2], is_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_transpose_backward_weights(
        weight.sizes(), grad_output, input, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups, output_mask[2], is_channels_last);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}
}

REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_transpose_stub, &mkldnn_convolution_transpose);
REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_transpose_backward_stub, &mkldnn_convolution_transpose_backward);

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise"),
      TORCH_FN(mkldnn_convolution_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise_.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary_));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      TORCH_FN(mkldnn_convolution_transpose_pointwise));
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise"),
      TORCH_FN(mkldnn_convolution_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise_.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary_));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      TORCH_FN(mkldnn_convolution_transpose_pointwise));
}

TORCH_LIBRARY_IMPL(mkldnn, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      TORCH_FN(mkldnn_convolution_transpose_pointwise_meta));
}
}}  // namespace at::native

#endif
