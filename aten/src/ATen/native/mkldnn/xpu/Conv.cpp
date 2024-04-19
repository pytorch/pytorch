#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/full.h>
#include <ATen/ops/neg.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <torch/library.h>

using namespace dnnl;
using namespace at::native;
using namespace at::native::onednn;

namespace at::native {
namespace xpu {
namespace impl {

struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;

  bool is_strided() const;
  bool is_dilated() const;
  bool is_padded() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  bool is_stride_nonpos() const;
  void view1d_as_2d();
  bool use_cpu_depthwise3x3_winograd(
      const at::Tensor& input,
      const at::Tensor& weight) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

std::ostream& operator<<(std::ostream& out, const ConvParams& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << IntArrayRef{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << IntArrayRef{params.output_padding}
      << "  groups = " << params.groups << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic << "}";
  return out;
}

bool ConvParams::is_strided() const {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

bool ConvParams::is_dilated() const {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

bool ConvParams::is_padded() const {
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

bool ConvParams::is_output_padding_neg() const {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_output_padding_big() const {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |=
        (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

bool ConvParams::is_padding_neg() const {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_stride_nonpos() const {
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

void ConvParams::view1d_as_2d() {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

bool ConvParams::use_cpu_depthwise3x3_winograd(
    const at::Tensor& input,
    const at::Tensor& weight) const {
  return false;
}

bool ConvParams::is_depthwise(const at::Tensor& input, const at::Tensor& weight)
    const {
  return !transposed && input.ndimension() == 4 && input.size(1) == groups &&
      groups > 1 && // no point if there is only a single group
      weight.size(0) % input.size(1) ==
      0; // output channels must be a multiple of input channels
}

static void check_shape_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ConvParams& params,
    bool input_is_mkldnn) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight.ndimension();
  std::vector<int64_t> weight_sizes(weight_dim);
  if ((weight_dim == k + 1) && input_is_mkldnn) {
    weight_sizes[0] = weight.size(0) * weight.size(1);
    std::copy_n(weight.sizes().cbegin() + 2, k - 1, weight_sizes.begin() + 1);
    weight_dim = k;
  } else {
    std::copy_n(weight.sizes().cbegin(), weight_dim, weight_sizes.begin());
  }
  int64_t groups = params.groups;
  auto padding = params.padding;
  auto output_padding = params.output_padding;
  auto stride = params.stride;
  auto dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(
      !params.is_output_padding_neg(),
      "negative output_padding is not supported");
  TORCH_CHECK(
      !params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      input.sizes(),
      " instead");
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(
        input.size(1) == (weight_sizes[1] * groups),
        "Given groups=",
        groups,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        (weight_sizes[1] * groups),
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    for (int i = 2; i < k; ++i) {
      input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(
        input_shape.size() == kernel_shape.size(),
        "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(
          0,
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }
  } else {
    TORCH_CHECK(
        input.size(1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");
  }
}

static at::Tensor view4d(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.ndimension() == 3,
      "expected 3D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.unsqueeze(2);
}

static at::Tensor view3d(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.ndimension() == 4,
      "expected 4D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.squeeze(2);
}

Attr get_onednn_conv_sum_attr(
    const Tensor& input_r,
    const Tensor& weight_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    Tensor& accumu,
    double scale,
    Tensor& output,
    bool& is_fused,
    Attr attr = Attr(),
    bool force_inplace = false) {
  is_fused = true;
  if (scale == 0.f)
    return attr;

  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_size(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  MemoryFormat mem_fmt = at::MemoryFormat::Contiguous;
  auto input_fmt = input_r.suggest_memory_format();
  auto input_is_cl = (input_fmt == at::MemoryFormat::ChannelsLast || input_fmt == at::MemoryFormat::ChannelsLast3d);
  auto weight_fmt = weight_r.suggest_memory_format();
  auto weight_is_cl = (weight_fmt == at::MemoryFormat::ChannelsLast || weight_fmt == at::MemoryFormat::ChannelsLast3d);

  bool propagate_channels_last = input_is_cl || weight_is_cl;
  if (propagate_channels_last)
    mem_fmt = get_cl_tag_by_ndim(ndim);

  Tensor out = at::empty(output_size, input_r.options().memory_format(mem_fmt));
  if (!onednn::binary_valid(out, accumu)) {
    is_fused = false;
    return attr;
  }

  // For post-sum and post-binary-add, onednn needs sum/binary scale=1.f
  // Thus we need the following transformation
  // conv(src, wei) + scale * accumu
  // scale * (1/scale * conv(src, wei) + sum (or binary))
  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  if (force_inplace) {
    // If sizes are the same, post sum is used.
    output = accumu;
    attr.append_post_sum(/* sum_scale */ 1.f);
  } else {
    // If sizes are different, post binary is used.
    attr.append_post_binary(attr.kind_with_binary_add, accumu);
  }

  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  return attr;
}

} // namespace impl

using namespace impl;

Tensor _convolution_out(
    Tensor& output_r,
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Attr attr,
    IntArrayRef pad_nd = IntArrayRef({})) {
  auto ndim = input_r.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution only supports 3D, 4D, 5D tensor");
  // get computation format for Conv/TransposedConv
  bool is_channels_last_suggested = use_channels_last_for_conv(input_r, weight_r, transposed_);

  Tensor input = input_r, weight = weight_r;
  // PyTorch does not support ChannelsLast1D case,
  // thus we need the transformation here
  if (ndim == 3) {
    input = view4d(input_r);
    weight = view4d(weight_r);
  }
  // ensure the input/weight/bias/output are congituous in desired format
  at::MemoryFormat mfmt = is_channels_last_suggested
      ? get_cl_tag_by_ndim(input.ndimension())
      : at::MemoryFormat::Contiguous;
  auto bias = bias_r.defined() ? bias_r.contiguous() : bias_r;
  input = input.contiguous(mfmt);
  weight = weight.contiguous(mfmt);

  auto k = weight.ndimension();
  if (k == input.ndimension() + 1) {
    k = input.ndimension();
  }
  int64_t dim = k - 2;
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  if (ndim == 3) {
    // PyTorch does not support ChannelsLast1D case,
    // thus we need the transformation here
    params.stride = stride_.vec();
    params.padding = padding_.vec();
    params.dilation = dilation_.vec();
    params.transposed = transposed_;
    params.output_padding = output_padding_.vec();
    params.groups = groups_;
    params.view1d_as_2d();
  } else {
    params.stride = expand_param_if_needed(stride_, "stride", dim);
    // PyTorch default Conv padding should be a single integer value
    // or a list of values to match the conv dimensions
    // conv2d, the number of padding values should be 1 or 2
    // conv3d, the number of padding values should be 1 or 3
    // the padding value will be padded into both side of Conv input (D, H, W)
    params.padding = expand_param_if_needed(padding_, "padding", dim);
    params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
    params.transposed = transposed_;
    params.output_padding =
        expand_param_if_needed(output_padding_, "output_padding", dim);
    params.groups = groups_;
  }
  check_shape_forward(input, weight, bias, params, true);

  Tensor output;
  if (transposed_) {
    // create output and propagate memory format
    if (!output_r.defined()) {
      auto dst_tz = deconv_dst_size(
          input.sizes(),
          weight.sizes(),
          params.padding,
          params.stride,
          params.dilation,
          params.output_padding,
          params.groups);
      output = at::empty(dst_tz, input.options(), mfmt);
    }

    onednn::deconvolution(
        output,
        input,
        weight,
        bias,
        params.stride,
        params.padding,
        params.output_padding,
        params.dilation,
        params.groups,
        attr);
  } else {
    // oneDNN supports padding the two sides of src with different values
    // the padding order should be front_top_left and back_bottom_right
    auto padding_front_top_left = params.padding;
    auto padding_back_bottom_right = params.padding;

    // PyTorch constant_pad_nd:
    // can pad different value to the two sides of Conv input (W, H, D)
    // (padding_left, padding_right,
    //  padding_top, padding_bottom,
    //  padding_front, padding_back)
    if (pad_nd.vec().size() > 0) {
      for (int i = 0; i < dim; ++i) {
        padding_front_top_left[i] += pad_nd[2 * dim - 2 * i - 2]; // 4, 2, 0
        padding_back_bottom_right[i] += pad_nd[2 * dim - 2 * i - 1]; // 5, 3, 1
      }
    }

    // create output and propagate memory format
    if (! output_r.defined()) {
      auto dst_tz = conv_dst_size(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding_front_top_left,
          padding_back_bottom_right,
          params.stride,
          params.dilation);
      output = at::empty(dst_tz, input.options(), mfmt);
    }
    onednn::convolution(
        output,
        input,
        weight,
        bias,
        padding_front_top_left,
        padding_back_bottom_right,
        params.stride,
        params.dilation,
        params.groups,
        attr);
  }

  if (ndim == 3) {
    output = view3d(output);
  }
  if (output_r.defined() && !output_r.is_same(output)) {
    output_r.copy_(output);
  } else {
    output_r = output;
  }
  return output_r;
}

Tensor _convolution(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Attr attr) {
  Tensor output_r;
  return _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_overrideable(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<at::Tensor>& bias_r_opt,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_) {
  c10::MaybeOwned<Tensor> bias_r_maybe_owned =
      at::borrow_from_optional_tensor(bias_r_opt);
  const Tensor& bias_r = *bias_r_maybe_owned;
  return _convolution(
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      Attr());
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto ndim = input.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution bwd only supports 3D, 4D, 5D tensor");
  TORCH_CHECK(
      grad_output.scalar_type() == ScalarType::Float ||
          grad_output.scalar_type() == ScalarType::BFloat16 ||
          grad_output.scalar_type() == ScalarType::Double ||
          grad_output.scalar_type() == ScalarType::Half,
      "so far only support float, bfloat16, half and double convolution backward in XPU backend, your data type is ",
      grad_output.scalar_type());

  bool is_channels_last_suggested = use_channels_last_for_conv(input, weight, transposed);

  Tensor grad_output_, input_, weight_;
  IntArrayRef stride_, padding_, dilation_, output_padding_;
  bool transposed_;
  int64_t groups_;
  ConvParams params;
  if (3 == ndim) {
    grad_output_ = view4d(grad_output);
    input_ = view4d(input);
    weight_ = view4d(weight);
    params.stride = stride.vec();
    params.padding = padding.vec();
    params.dilation = dilation.vec();
    params.transposed = transposed;
    params.output_padding = output_padding.vec();
    params.groups = groups;
    params.view1d_as_2d();
    stride_ = params.stride;
    padding_ = params.padding;
    dilation_ = params.dilation;
    transposed_ = params.transposed;
    output_padding_ = params.output_padding;
    groups_ = params.groups;
  } else {
    grad_output_ = grad_output;
    input_ = input;
    weight_ = weight;
    stride_ = stride;
    padding_ = padding;
    dilation_ = dilation;
    transposed_ = transposed;
    output_padding_ = output_padding;
    groups_ = groups;
  }

  // ensure the tensors are contiguous
  auto mfmt = is_channels_last_suggested ? get_cl_tag_by_ndim(input_.ndimension())
      : at::MemoryFormat::Contiguous;
  grad_output_ =  grad_output_.contiguous(mfmt);
  weight_ = weight_.contiguous(mfmt);
  input_ = input_.contiguous(mfmt);

  auto opt = grad_output_.options();
  Tensor grad_input = at::empty(input_.sizes(), opt, mfmt);
  Tensor grad_weight = at::empty(weight_.sizes(), opt, mfmt);
  Tensor grad_bias;
  if (output_mask[2])
    grad_bias = at::empty({grad_output_.size(1)}, opt);

  if (output_mask[0]) {
    if (input.numel() > 0) {
      if (transposed_) {
        onednn::deconvolution_backward_data(
            grad_input,
            grad_output_,
            weight_,
            stride_,
            padding_,
            dilation_,
            groups_,
            output_mask[2]);
      } else {
        onednn::convolution_backward_data(
            grad_input,
            grad_output_,
            weight_,
            padding_,
            padding_,
            stride_,
            dilation_,
            groups_,
            output_mask[2]);
      }
    }
  }
  if (output_mask[1] || output_mask[2]) {
    if (input.numel() > 0) {
      if (transposed_) {
        onednn::deconvolution_backward_weights(
            grad_weight,
            grad_bias,
            grad_output_,
            input_,
            stride_,
            padding_,
            dilation_,
            groups_);
      } else {
        onednn::convolution_backward_weights(
            grad_weight,
            grad_bias,
            grad_output_,
            input_,
            weight_.sizes(),
            padding_,
            padding_,
            stride_,
            dilation_,
            groups_);
      }
    }
  }

  if (3 == ndim) {
    if (output_mask[0])
      grad_input = view3d(grad_input);
    grad_weight = view3d(grad_weight);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

TORCH_LIBRARY_IMPL(aten, XPU, m){
  m.impl("convolution_overrideable", TORCH_FN(convolution_overrideable));
  m.impl("convolution_backward_overrideable", TORCH_FN(convolution_backward_overrideable));
}

} // namespace xpu
} // namespace at::native
