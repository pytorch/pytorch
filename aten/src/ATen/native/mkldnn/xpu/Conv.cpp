#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <vector>

#include <oneDNN/oneDNN.h>
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/interned_strings.h"
#include "ATen/ops/full.h"
#include "ATen/ops/neg.h"
#include "FusionUtils.h"
#include "c10/core/Scalar.h"
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "comm/ParamUtils.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha);
Tensor mul_add_scalar(
    const Tensor& self,
    Scalar other,
    const Tensor& accumu,
    Scalar alpha);

namespace {
// For few binary ops do not have unify interface. In order to simplify the code
// and for better abstraction, we redefine those binary functino in anonymous
// namespace.
#define DEFINE_BINARY_FUNC(func)                          \
  static Tensor func(Tensor& src, const Tensor& binary) { \
    return AtenIpexTypeXPU::func(src, binary);            \
  }                                                       \
  static Tensor func(Tensor& src, const Scalar& binary) { \
    return AtenIpexTypeXPU::func(src, binary);            \
  }

DEFINE_BINARY_FUNC(mul)
DEFINE_BINARY_FUNC(div)
DEFINE_BINARY_FUNC(eq)
DEFINE_BINARY_FUNC(ne)
DEFINE_BINARY_FUNC(gt)
DEFINE_BINARY_FUNC(ge)
DEFINE_BINARY_FUNC(le)
DEFINE_BINARY_FUNC(lt)
// AtenIpexTypeXPU namespace have not define the fmin fmax, we call those two
// function on at namespace.
static Tensor min(Tensor& src, const Tensor& binary) {
  return at::fmin(src, binary);
}
static Tensor max(Tensor& src, const Tensor& binary) {
  return at::fmax(src, binary);
}
static Tensor min(Tensor& src, const Scalar& binary) {
  Tensor binary_tensor = at::full_like(src, binary);
  return at::fmin(src, binary_tensor);
}
static Tensor max(Tensor& src, const Scalar& binary) {
  Tensor binary_tensor = at::full_like(src, binary);
  return at::fmax(src, binary_tensor);
}

/* IPEX_CONV_DEFINATION
This macro is used to generate the defination of conv2d, _convolution related
post-op functions in a convinent way. It can only be used when post-op's name in
function defination is exactly the same as the name in Attr's defined post-ops,
and no any extra parameters is brought in compared to the original convolution
signiture.
*/

#define IPEX_CONV_DEFINATION(op)                                   \
  Tensor convolution_##op(                                         \
      const Tensor& input,                                         \
      const Tensor& weight,                                        \
      const c10::optional<Tensor>& bias,                           \
      std::vector<int64_t> stride_,                                \
      std::vector<int64_t> padding_,                               \
      std::vector<int64_t> dilation_,                              \
      int64_t groups_) {                                           \
    Attr att;                                                      \
    att.append_post_eltwise(1.0f, 0.0f, 0.0f, att.kind_with_##op); \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor(); \
    return _convolution(                                           \
        input,                                                     \
        weight,                                                    \
        bias_,                                                     \
        stride_,                                                   \
        padding_,                                                  \
        dilation_,                                                 \
        false,                                                     \
        {{0, 0}},                                                  \
        groups_,                                                   \
        att);                                                      \
  }                                                                \
                                                                   \
  Tensor _convolution_##op(                                        \
      const Tensor& input,                                         \
      const Tensor& weight,                                        \
      const c10::optional<Tensor>& bias,                           \
      std::vector<int64_t> stride_,                                \
      std::vector<int64_t> padding_,                               \
      std::vector<int64_t> dilation_,                              \
      bool transposed,                                             \
      std::vector<int64_t> output_padding_,                        \
      int64_t groups,                                              \
      bool benchmark,                                              \
      bool deterministic,                                          \
      bool cudnn_enabled,                                          \
      bool allow_tf32) {                                           \
    Attr att;                                                      \
    att.append_post_eltwise(1.0f, 0.0f, 0.0f, att.kind_with_##op); \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor(); \
    return _convolution(                                           \
        input,                                                     \
        weight,                                                    \
        bias_,                                                     \
        stride_,                                                   \
        padding_,                                                  \
        dilation_,                                                 \
        transposed,                                                \
        output_padding_,                                           \
        groups,                                                    \
        att);                                                      \
  }

#define IPEX_CONV_BINARY_DEFINATION(op)                                 \
  Tensor _convolution_binary_##op(                                      \
      const Tensor& input_r,                                            \
      const Tensor& weight_r,                                           \
      const c10::optional<Tensor>& bias,                                \
      std::vector<int64_t> stride_,                                     \
      std::vector<int64_t> padding_,                                    \
      std::vector<int64_t> dilation_,                                   \
      bool transposed,                                                  \
      std::vector<int64_t> output_padding_,                             \
      int64_t groups_,                                                  \
      bool benchmark,                                                   \
      bool deterministic,                                               \
      bool cudnn_enabled,                                               \
      bool allow_tf32,                                                  \
      const Tensor& binary) {                                           \
    Attr attr;                                                          \
    auto ndim = input_r.ndimension();                                   \
    auto output_size = conv_dst_tz(                                     \
        ndim,                                                           \
        input_r.sizes(),                                                \
        weight_r.sizes(),                                               \
        padding_,                                                       \
        padding_,                                                       \
        stride_,                                                        \
        dilation_);                                                     \
    Tensor out = at::empty(output_size, input_r.options());             \
    bool binary_enabled = xpu::oneDNN::binary_valid(out, binary);       \
    if (binary_enabled) {                                               \
      attr.append_post_binary(attr.kind_with_binary_##op, binary);      \
    }                                                                   \
    Tensor output_r;                                                    \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();      \
    Tensor ret = _convolution_out(                                      \
        output_r,                                                       \
        input_r,                                                        \
        weight_r,                                                       \
        bias_,                                                          \
        stride_,                                                        \
        padding_,                                                       \
        dilation_,                                                      \
        false,                                                          \
        {{0, 0}},                                                       \
        groups_,                                                        \
        attr);                                                          \
    if (!binary_enabled) {                                              \
      return op(ret, binary);                                           \
    }                                                                   \
    return ret;                                                         \
  }                                                                     \
  Tensor convolution_binary_##op(                                       \
      const Tensor& input_r,                                            \
      const Tensor& weight_r,                                           \
      const c10::optional<Tensor>& bias,                                \
      std::vector<int64_t> stride_,                                     \
      std::vector<int64_t> padding_,                                    \
      std::vector<int64_t> dilation_,                                   \
      int64_t groups_,                                                  \
      const Tensor& binary) {                                           \
    Attr attr;                                                          \
    auto ndim = input_r.ndimension();                                   \
    auto output_size = conv_dst_tz(                                     \
        ndim,                                                           \
        input_r.sizes(),                                                \
        weight_r.sizes(),                                               \
        padding_,                                                       \
        padding_,                                                       \
        stride_,                                                        \
        dilation_);                                                     \
    Tensor out = at::empty(output_size, input_r.options());             \
    bool binary_enabled = xpu::oneDNN::binary_valid(out, binary);       \
    if (binary_enabled) {                                               \
      attr.append_post_binary(attr.kind_with_binary_##op, binary);      \
    }                                                                   \
    Tensor output_r;                                                    \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();      \
    Tensor ret = _convolution_out(                                      \
        output_r,                                                       \
        input_r,                                                        \
        weight_r,                                                       \
        bias_,                                                          \
        stride_,                                                        \
        padding_,                                                       \
        dilation_,                                                      \
        false,                                                          \
        {{0, 0}},                                                       \
        groups_,                                                        \
        attr);                                                          \
    if (!binary_enabled) {                                              \
      return op(ret, binary);                                           \
    }                                                                   \
    return ret;                                                         \
  }                                                                     \
  Tensor _convolution_binary_##op##_scalar(                             \
      const Tensor& input_r,                                            \
      const Tensor& weight_r,                                           \
      const c10::optional<Tensor>& bias,                                \
      std::vector<int64_t> stride_,                                     \
      std::vector<int64_t> padding_,                                    \
      std::vector<int64_t> dilation_,                                   \
      bool transposed,                                                  \
      std::vector<int64_t> output_padding_,                             \
      int64_t groups_,                                                  \
      bool benchmark,                                                   \
      bool deterministic,                                               \
      bool cudnn_enabled,                                               \
      bool allow_tf32,                                                  \
      const Scalar& binary) {                                           \
    Attr attr;                                                          \
    auto ndim = input_r.ndimension();                                   \
    auto output_size = conv_dst_tz(                                     \
        ndim,                                                           \
        input_r.sizes(),                                                \
        weight_r.sizes(),                                               \
        padding_,                                                       \
        padding_,                                                       \
        stride_,                                                        \
        dilation_);                                                     \
    Tensor out = at::empty(output_size, input_r.options());             \
    Tensor binary_tensor = at::full_like(out, binary);                  \
    attr.append_post_binary(attr.kind_with_binary_##op, binary_tensor); \
    Tensor output_r;                                                    \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();      \
    Tensor ret = _convolution_out(                                      \
        output_r,                                                       \
        input_r,                                                        \
        weight_r,                                                       \
        bias_,                                                          \
        stride_,                                                        \
        padding_,                                                       \
        dilation_,                                                      \
        false,                                                          \
        {{0, 0}},                                                       \
        groups_,                                                        \
        attr);                                                          \
    return ret;                                                         \
  }                                                                     \
  Tensor convolution_binary_##op##_scalar(                              \
      const Tensor& input_r,                                            \
      const Tensor& weight_r,                                           \
      const c10::optional<Tensor>& bias,                                \
      std::vector<int64_t> stride_,                                     \
      std::vector<int64_t> padding_,                                    \
      std::vector<int64_t> dilation_,                                   \
      int64_t groups_,                                                  \
      const Scalar& binary) {                                           \
    Attr attr;                                                          \
    auto ndim = input_r.ndimension();                                   \
    auto output_size = conv_dst_tz(                                     \
        ndim,                                                           \
        input_r.sizes(),                                                \
        weight_r.sizes(),                                               \
        padding_,                                                       \
        padding_,                                                       \
        stride_,                                                        \
        dilation_);                                                     \
    Tensor out = at::empty(output_size, input_r.options());             \
    Tensor binary_tensor = at::full_like(out, binary);                  \
    attr.append_post_binary(attr.kind_with_binary_##op, binary_tensor); \
    Tensor output_r;                                                    \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();      \
    Tensor ret = _convolution_out(                                      \
        output_r,                                                       \
        input_r,                                                        \
        weight_r,                                                       \
        bias_,                                                          \
        stride_,                                                        \
        padding_,                                                       \
        dilation_,                                                      \
        false,                                                          \
        {{0, 0}},                                                       \
        groups_,                                                        \
        attr);                                                          \
    return ret;                                                         \
  }

} // namespace
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
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  MemoryFormat mem_fmt = at::MemoryFormat::Contiguous;
  bool propagate_channels_last =
      is_smf_channels_last(input_r) || is_smf_channels_last(weight_r);
  if (propagate_channels_last)
    mem_fmt = get_cl_tag_by_ndim(ndim);

  Tensor out = at::empty(output_size, input_r.options().memory_format(mem_fmt));
  if (!xpu::oneDNN::binary_valid(out, accumu)) {
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

  accumu = contiguous_if_needed(accumu, mem_fmt);
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
  auto memory_layout_for_conv =
      get_memory_layout_for_conv(input_r, weight_r, transposed_);
  bool is_channels_last_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  Tensor input = input_r, weight = weight_r;
  // PyTorch does not support ChannelsLast1D case,
  // thus we need the transformation here
  if (ndim == 3 && !is_onednn_layout_suggested) {
    input = view4d(input_r);
    weight = view4d(weight_r);
  }
  // ensure the input/weight/bias/output are congituous in desired format
  at::MemoryFormat mfmt = is_channels_last_suggested
      ? get_cl_tag_by_ndim(input.ndimension())
      : at::MemoryFormat::Contiguous;
  input = contiguous_if_needed(input, mfmt);
  weight = contiguous_if_needed(weight, mfmt);
  auto bias = bias_r.defined() ? bias_r.contiguous() : bias_r;

  auto k = weight.ndimension();
  if (k == input.ndimension() + 1) {
    k = input.ndimension();
  }
  int64_t dim = k - 2;
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  if (ndim == 3 && !is_onednn_layout_suggested) {
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
    if (output_r.defined()) {
      output = contiguous_if_needed(output_r, mfmt);
    } else {
      auto dst_tz = deconv_dst_tz(
          input.sizes(),
          weight.sizes(),
          params.padding,
          params.stride,
          params.dilation,
          params.output_padding,
          params.groups);
      output = at::empty(dst_tz, input.options(), mfmt);
    }
    xpu::oneDNN::deconvolution(
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
    if (output_r.defined()) {
      output = contiguous_if_needed(output_r, mfmt);
    } else {
      auto dst_tz = conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding_front_top_left,
          padding_back_bottom_right,
          params.stride,
          params.dilation);
      output = at::empty(dst_tz, input.options(), mfmt);
    }
    output = xpu::oneDNN::convolution(
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

  if (ndim == 3 && !is_onednn_layout_suggested) {
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

Tensor pad_convolution(
    const Tensor& input_r,
    IntArrayRef pad_nd,
    Scalar value,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_) {
  // oneDNN only support padding value with 0
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  if (value.to<float>() != 0.0) {
    auto padded_input = at::constant_pad_nd(input_r, pad_nd, value);
    Tensor output_r;
    return _convolution_out(
        output_r,
        padded_input,
        weight_r,
        bias_r,
        stride_,
        padding_,
        dilation_,
        false,
        {{0, 0}},
        groups_,
        Attr());
  }

  Tensor output_r;
  return _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      Attr(),
      pad_nd);
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
          grad_output.scalar_type() == ScalarType::Double,
      "so far only support float, bfloat16 and double convolution backward in XPU backend, your data type is ",
      grad_output.scalar_type());

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(input, weight, transposed);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;
  bool is_channels_last_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;

  Tensor grad_output_, input_, weight_;
  IntArrayRef stride_, padding_, dilation_, output_padding_;
  bool transposed_;
  int64_t groups_;
  ConvParams params;
  if (3 == ndim && !is_onednn_layout_suggested) {
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
  auto mfmt = is_channels_last_suggested
      ? get_cl_tag_by_ndim(input_.ndimension())
      : at::MemoryFormat::Contiguous;
  input_ = contiguous_if_needed(input_, mfmt);
  weight_ = contiguous_if_needed(weight_, mfmt);
  grad_output_ = contiguous_if_needed(grad_output_, mfmt);

  auto opt = grad_output_.options();
  Tensor grad_input = at::empty(input_.sizes(), opt, mfmt);
  Tensor grad_weight = at::empty(weight_.sizes(), opt, mfmt);
  Tensor grad_bias;
  if (output_mask[2])
    grad_bias = at::empty({grad_output_.size(1)}, opt);

  if (output_mask[0]) {
    if (input.numel() > 0) {
      if (transposed_) {
        xpu::oneDNN::deconvolution_backward_data(
            grad_input,
            grad_output_,
            weight_,
            stride_,
            padding_,
            dilation_,
            groups_,
            output_mask[2]);
      } else {
        xpu::oneDNN::convolution_backward_data(
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
        xpu::oneDNN::deconvolution_backward_weights(
            grad_weight,
            grad_bias,
            grad_output_,
            input_,
            stride_,
            padding_,
            dilation_,
            groups_);
      } else {
        xpu::oneDNN::convolution_backward_weights(
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

  if (3 == ndim && !is_onednn_layout_suggested) {
    if (output_mask[0])
      grad_input = view3d(grad_input);
    grad_weight = view3d(grad_weight);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

// It is recommand to define the post-op function name follow the pytorch's
// inner op rather than oneDNN post op name, for example, we can find pytorch's
// silu op named swish in oneDNN. For clarity, we use all the pytorch's op name
// in function defination. Therefore, for the fusion of convolution + silu, we
// name the funciton as convolution_silu.
IPEX_CONV_DEFINATION(sqrt)
IPEX_CONV_DEFINATION(abs)
IPEX_CONV_DEFINATION(tanh)
IPEX_CONV_DEFINATION(square)
IPEX_CONV_DEFINATION(exp)
IPEX_CONV_DEFINATION(log)
IPEX_CONV_DEFINATION(round)
IPEX_CONV_DEFINATION(sigmoid)
IPEX_CONV_DEFINATION(relu)
IPEX_CONV_DEFINATION(mish)

Tensor convolution_hardswish(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.f / 6.f, 1.f / 2.f, att.kind_with_hardswish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_hardswish(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.f / 6.f, 1.f / 2.f, att.kind_with_hardswish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(1.0f, -1.0f, 0.0f, att.kind_with_soft_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(1.0f, -1.0, 0.0f, att.kind_with_soft_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    c10::string_view approximate) {
  Attr att;
  algorithm algo;
  if (approximate == "none") {
    algo = att.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = att.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  att.append_post_eltwise(1.0f, 0.0f, 0.0f, algo);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    c10::string_view approximate) {
  Attr att;
  algorithm algo;
  if (approximate == "none") {
    algo = att.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = att.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  att.append_post_eltwise(1.0f, 0.0f, 0.0f, algo);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor _convolution_mish_compound(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed_,
    std::vector<int64_t> output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar beta,
    Scalar threshold) {
  return _convolution_mish(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      benchmark,
      deterministic,
      cudnn_enabled,
      allow_tf32);
}

Tensor convolution_mish_compound(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar beta,
    Scalar threshold) {
  return convolution_mish(
      input, weight, bias, stride_, padding_, dilation_, groups_);
}

Tensor _convolution_mish_compound_add(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed_,
    std::vector<int64_t> output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar beta,
    Scalar threshold,
    Tensor accumu,
    Scalar scale) {
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution mish fusion with mish scale equals to 1, alpha equal to 1");
  Attr attr;
  attr.append_post_eltwise(
      /* mish_scale */ 1.0,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_mish);
  bool is_fused = true;
  Tensor output;
  attr = get_onednn_conv_sum_attr(
      input,
      weight,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused,
      attr);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor res = _convolution_out(
      output,
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
  if (!is_fused) {
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
  }
  return res;
}

Tensor convolution_mish_compound_add(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar beta,
    Scalar threshold,
    Tensor accumu,
    Scalar scale) {
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution mish fusion with mish scale equals to 1, alpha equal to 1");
  Attr attr;
  attr.append_post_eltwise(
      /* mish_scale */ 1.0,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_mish);
  bool is_fused = true;
  Tensor output;
  attr = get_onednn_conv_sum_attr(
      input,
      weight,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused,
      attr);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor res = _convolution_out(
      output,
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {0, 0},
      groups_,
      attr);
  if (!is_fused) {
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
  }
  return res;
}

// IPEX_CONV_BINARY_DEFINATION(add)
IPEX_CONV_BINARY_DEFINATION(mul)
// IPEX_CONV_BINARY_DEFINATION(sub)
IPEX_CONV_BINARY_DEFINATION(div)
IPEX_CONV_BINARY_DEFINATION(max)
IPEX_CONV_BINARY_DEFINATION(min)
IPEX_CONV_BINARY_DEFINATION(eq)
IPEX_CONV_BINARY_DEFINATION(ne)
IPEX_CONV_BINARY_DEFINATION(ge)
IPEX_CONV_BINARY_DEFINATION(gt)
IPEX_CONV_BINARY_DEFINATION(le)
IPEX_CONV_BINARY_DEFINATION(lt)

Tensor convolution_silu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_swish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_silu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_swish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}
Tensor convolution_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(
      1.0f, 1.0f / 6., 1.0f / 2., att.kind_with_hardsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(
      1.0f, 1.0f / 6., 1.0f / 2., att.kind_with_hardsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar exponent) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, exponent.toFloat(), att.kind_with_pow);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar exponent) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, exponent.toFloat(), att.kind_with_pow);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar negative_slope) {
  Attr att;
  att.append_post_eltwise(
      1.0f, negative_slope.toFloat(), 0.f, att.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar negative_slope) {
  Attr att;
  att.append_post_eltwise(
      1.0f, negative_slope.toFloat(), 0.f, att.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar minval,
    Scalar maxval) {
  Attr att;
  att.append_post_eltwise(
      1.0f, minval.toFloat(), maxval.toFloat(), att.kind_with_clip);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar minval,
    Scalar maxval) {
  Attr att;
  att.append_post_eltwise(
      1.0f, minval.toFloat(), maxval.toFloat(), att.kind_with_clip);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  AT_ASSERT(
      scale.toFloat() == 1.0f && input_scale.toFloat() == 1.0f,
      "elu's scale and input scale can only be 1.f in jit fusion");
  Attr att;
  att.append_post_eltwise(1.0f, alpha.toFloat(), 1.0f, att.kind_with_elu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  AT_ASSERT(
      scale.toFloat() == 1.0f && input_scale.toFloat() == 1.0f,
      "elu's scale and input scale can only be 1.f in jit fusion");
  Attr att;
  att.append_post_eltwise(1.0f, alpha.toFloat(), 1.0f, att.kind_with_elu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_sum_inplace(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused,
      Attr(),
      true);
  Tensor res = _convolution_out(
      output,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_(res, accumu, 1.f);
  return res;
}

Tensor convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused);
  Tensor res = _convolution_out(
      output,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
  return res;
}

Tensor convolution_sum_scalar(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Scalar& binary,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  Tensor binary_tensor = at::full_like(out, binary);
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      binary_tensor,
      scale.to<float>(),
      output,
      is_fused);
  Tensor res = _convolution_out(
      output,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_out(res, binary_tensor, 1.f, binary_tensor);
  return res;
}

Tensor _convolution_sum_scalar(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    const Scalar& binary,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  Tensor binary_tensor = at::full_like(out, binary);
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      binary_tensor,
      scale.to<float>(),
      output,
      is_fused);
  Tensor res = _convolution_out(
      output,
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
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_out(res, binary_tensor, 1.f, binary_tensor);
  return res;
}

Tensor _convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused);
  Tensor res = _convolution_out(
      output,
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
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
  return res;
}

Tensor _convolution_sum_inplace(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused,
      Attr(),
      true);
  Tensor res = _convolution_out(
      output,
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
  if (!is_fused)
    res = at::AtenIpexTypeXPU::add_(res, accumu, 1.f);
  return res;
}

Tensor _convolution_binary_sub(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& binary,
    Scalar scale) {
  return _convolution_sum(
      input_r,
      weight_r,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups_,
      benchmark,
      deterministic,
      cudnn_enabled,
      allow_tf32,
      binary,
      -scale);
}

Tensor _convolution_binary_sub_scalar(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    const Scalar& binary,
    Scalar scale) {
  return _convolution_sum_scalar(
      input_r,
      weight_r,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups_,
      benchmark,
      deterministic,
      cudnn_enabled,
      allow_tf32,
      binary,
      -scale);
}

Tensor convolution_binary_sub(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Tensor& binary,
    Scalar scale) {
  return convolution_sum(
      input_r,
      weight_r,
      bias,
      stride_,
      padding_,
      dilation_,
      groups_,
      binary,
      -scale);
}

Tensor convolution_binary_sub_scalar(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    const Scalar& binary,
    Scalar scale) {
  return convolution_sum_scalar(
      input_r,
      weight_r,
      bias,
      stride_,
      padding_,
      dilation_,
      groups_,
      binary,
      -scale);
}

Tensor convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused);
  if (is_fused) {
    attr.append_post_eltwise( // append post relu
        /* relu_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  }
  Tensor res = _convolution_out(
      output,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!is_fused) {
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
    res = at::AtenIpexTypeXPU::relu_(res);
  }
  return res;
}

Tensor _convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale) {
  bool is_fused;
  Tensor output;
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Attr attr = get_onednn_conv_sum_attr(
      input_r,
      weight_r,
      stride_,
      padding_,
      dilation_,
      accumu,
      scale.to<float>(),
      output,
      is_fused);
  if (is_fused) {
    attr.append_post_eltwise( // append post relu
        /* relu_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  }
  Tensor res = _convolution_out(
      accumu,
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
  if (!is_fused) {
    res = at::AtenIpexTypeXPU::add_out(res, accumu, 1.f, accumu);
    res = at::AtenIpexTypeXPU::relu_(res);
  }
  return res;
}

Tensor _convolution_binary_mul_add(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor binary_mul,
    Tensor binary_add,
    Scalar scale) {
  Attr attr;
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  // TODO(ganyi): This method is not very clean and clever, refine the code in
  // the future.
  bool mul = false, add = false;
  Tensor out = at::empty(output_size, input_r.options());
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    return AtenIpexTypeXPU::mul_add(ret, binary_mul, binary_add, scale);
  } else {
    return AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
}

Tensor convolution_binary_mul_add(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary_mul,
    Tensor& binary_add,
    Scalar scale) {
  Attr attr;
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  // TODO(ganyi): This method is not very clean and clever, refine the code in
  // the future.
  bool mul = false, add = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    return AtenIpexTypeXPU::mul_add(out, binary_mul, binary_add, scale);
  } else {
    return AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
}

Tensor convolution_sigmoid_binary_mul(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary_mul) {
  Attr attr;
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor bias_r = bias.has_value() ? bias.value() : at::Tensor();
  Tensor out = at::empty(output_size, input_r.options());
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  bool mul = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!mul) {
    return AtenIpexTypeXPU::mul(ret, binary_mul);
  }
  return ret;
}

Tensor convolution_sigmoid_binary_mul_add(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary_mul,
    Tensor& binary_add,
    Scalar scale) {
  Attr attr;
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  bool mul = false, add = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    return AtenIpexTypeXPU::mul_add(out, binary_mul, binary_add, scale);
  } else {
    return AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
}

Tensor convolution_sigmoid_binary_mul_add_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary_mul,
    Tensor& binary_add,
    Scalar scale) {
  Attr attr;
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  bool mul = false, add = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  if (add)
    attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    ret = AtenIpexTypeXPU::mul_add(out, binary_mul, binary_add, scale);
  } else {
    ret = AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
  return AtenIpexTypeXPU::relu(ret);
}

Tensor _convolution_sigmoid_binary_mul(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor binary_mul) {
  Attr attr;
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  bool mul = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  Tensor output_r;
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (!mul) {
    return AtenIpexTypeXPU::mul(ret, binary_mul);
  }
  return ret;
}

Tensor _convolution_sigmoid_binary_mul_add(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor binary_mul,
    Tensor binary_add,
    Scalar scale) {
  Attr attr;
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  bool mul = false, add = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    return AtenIpexTypeXPU::mul_add(ret, binary_mul, binary_add, scale);
  } else {
    return AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
}

Tensor _convolution_sigmoid_binary_mul_add_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor binary_mul,
    Tensor binary_add,
    Scalar scale) {
  Attr attr;
  attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_sigmoid);
  auto ndim = input_r.ndimension();
  auto output_size = conv_dst_tz(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  Tensor out = at::empty(output_size, input_r.options());
  bool mul = false, add = false;
  if (xpu::oneDNN::binary_valid(out, binary_mul)) {
    attr.append_post_binary(attr.kind_with_binary_mul, binary_mul);
    mul = true;
  }
  if (mul) {
    attr = get_onednn_conv_sum_attr(
        input_r,
        weight_r,
        stride_,
        padding_,
        dilation_,
        binary_add,
        scale.toFloat(),
        out,
        add,
        attr);
  }
  if (add)
    attr.append_post_eltwise(1.0, 0.0, 0.0, attr.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  Tensor ret = _convolution_out(
      out,
      input_r,
      weight_r,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  if (mul && add)
    return ret;
  else if (!mul) {
    ret = AtenIpexTypeXPU::mul_add(ret, binary_mul, binary_add, scale);
  } else {
    ret = AtenIpexTypeXPU::add(ret, binary_add, scale);
  }
  return AtenIpexTypeXPU::relu(ret);
}

Tensor onednn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_,
    IntArrayRef padding_,
    IntArrayRef stride_,
    IntArrayRef dilation_,
    int64_t groups_,
    c10::string_view attr = "none",
    torch::List<c10::optional<at::Scalar>> scalars =
        torch::List<c10::optional<at::Scalar>>(),
    c10::optional<c10::string_view> algorithm = c10::nullopt) {
  Attr att;
  att = construct_unary_attr(attr, scalars, algorithm, att);
  const Tensor bias = bias_.has_value() ? bias_.value() : at::Tensor();
  auto dim = input.ndimension();
  std::vector<int64_t> output_padding = {0};

  return _convolution(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      /*transposed*/ false,
      output_padding,
      groups_,
      att);
}

Tensor convolution_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm) {
  return onednn_convolution(
      input_t,
      weight_t,
      bias_opt,
      padding,
      stride,
      dilation,
      groups,
      attr,
      scalars,
      algorithm);
}

Tensor convolution_pointwise_meta(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm) {
  // oneDNN supports padding the two sides of src with different values
  // the padding order should be front_top_left and back_bottom_right
  auto padding_front_top_left = padding.vec();
  auto padding_back_bottom_right = padding.vec();

  auto k = weight_t.ndimension();
  if (k == input_t.ndimension() + 1) {
    k = input_t.ndimension();
  }
  int64_t dim = k - 2;

  auto dst_tz = conv_dst_tz(
      input_t.ndimension(),
      input_t.sizes(),
      weight_t.sizes(),
      padding,
      padding,
      stride,
      dilation);
  Tensor output = at::empty(dst_tz, input_t.options());
  return output;
}

Tensor convolution_pointwise_binary(
    const Tensor& input_t,
    Tensor& other_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view binary_attr,
    c10::optional<at::Scalar> alpha,
    c10::optional<c10::string_view> unary_attr,
    torch::List<c10::optional<at::Scalar>> unary_scalars,
    c10::optional<c10::string_view> unary_algorithm) {
  Tensor output;
  Tensor bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
  // Step1: Construct binary attr
  Attr attr;
  bool is_fused = true;
  if (binary_attr != "add") {
    attr = construct_binary_attr(binary_attr, alpha, other_t, attr);
  } else {
    attr = get_onednn_conv_sum_attr(
        input_t,
        weight_t,
        stride,
        padding,
        dilation,
        other_t,
        alpha.has_value() ? alpha.value().toFloat() : 1.f,
        output,
        /*is_fused*/ is_fused,
        attr);
  }

  // Step2: Append unary attr
  if (unary_attr.has_value())
    attr = construct_unary_attr(
        unary_attr.value(), unary_scalars, unary_algorithm, attr);

  Tensor res = _convolution_out(
      output,
      input_t,
      weight_t,
      bias,
      stride,
      padding,
      dilation,
      /*transpoced*/ false,
      {{0, 0}},
      groups,
      attr);

  // Step3: Run conv
  return res;
}

Tensor convolution_pointwise_binary_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view binary_attr,
    c10::optional<at::Scalar> alpha,
    c10::optional<c10::string_view> unary_attr,
    torch::List<c10::optional<at::Scalar>> unary_scalars,
    c10::optional<c10::string_view> unary_algorithm) {
  Tensor output;
  Tensor bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
  // Step1: Construct binary attr
  Attr attr;
  bool is_fused = true;
  if (binary_attr != "add") {
    attr = construct_binary_attr(binary_attr, alpha, other_t, attr);
  } else {
    attr = get_onednn_conv_sum_attr(
        input_t,
        weight_t,
        stride,
        padding,
        dilation,
        other_t,
        alpha.has_value() ? alpha.value().toFloat() : 1.f,
        output,
        /*is_fused*/ is_fused,
        attr,
        /*force_inplace*/ true);
  }

  // Step2: Append unary attr
  if (unary_attr.has_value())
    attr = construct_unary_attr(
        unary_attr.value(), unary_scalars, unary_algorithm, attr);

  Tensor res = _convolution_out(
      output,
      input_t,
      weight_t,
      bias,
      stride,
      padding,
      dilation,
      /*transpoced*/ false,
      {{0, 0}},
      groups,
      attr);

  // Step3: Run conv
  return res;
}

#define IPEX_OP_REGISTER_CONVOLUTION(op, ...)                    \
  IPEX_OP_REGISTER("conv2d_" #op __VA_ARGS__, convolution_##op); \
  IPEX_OP_REGISTER("_convolution_" #op __VA_ARGS__, _convolution_##op);

#define IPEX_OP_REGISTER_CONVOLUTION_SCALAR(op)                         \
  IPEX_OP_REGISTER("conv2d_" #op ".Scalar", convolution_##op##_scalar); \
  IPEX_OP_REGISTER("_convolution_" #op ".Scalar", _convolution_##op##_scalar);

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_CONVOLUTION(sigmoid);
  IPEX_OP_REGISTER_CONVOLUTION(relu);
  IPEX_OP_REGISTER_CONVOLUTION(sqrt);
  IPEX_OP_REGISTER_CONVOLUTION(abs);
  IPEX_OP_REGISTER_CONVOLUTION(tanh);
  IPEX_OP_REGISTER_CONVOLUTION(square);
  IPEX_OP_REGISTER_CONVOLUTION(exp);
  IPEX_OP_REGISTER_CONVOLUTION(log);
  IPEX_OP_REGISTER_CONVOLUTION(round);
  IPEX_OP_REGISTER_CONVOLUTION(log_sigmoid);
  IPEX_OP_REGISTER_CONVOLUTION(hardswish);
  IPEX_OP_REGISTER_CONVOLUTION(mish);
  IPEX_OP_REGISTER_CONVOLUTION(silu);
  IPEX_OP_REGISTER_CONVOLUTION(hardsigmoid);
  IPEX_OP_REGISTER_CONVOLUTION(gelu);
  IPEX_OP_REGISTER_CONVOLUTION(leaky_relu);
  IPEX_OP_REGISTER_CONVOLUTION(pow);
  IPEX_OP_REGISTER_CONVOLUTION(hardtanh);
  IPEX_OP_REGISTER_CONVOLUTION(elu);
  IPEX_OP_REGISTER_CONVOLUTION(sum);
  IPEX_OP_REGISTER_CONVOLUTION(sum_inplace);
  IPEX_OP_REGISTER_CONVOLUTION(sum_relu);
  IPEX_OP_REGISTER_CONVOLUTION(mish_compound);
  IPEX_OP_REGISTER_CONVOLUTION(mish_compound_add);
  IPEX_OP_REGISTER("pad_conv2d", pad_convolution);
  IPEX_OP_REGISTER_CONVOLUTION(binary_sub);
  IPEX_OP_REGISTER_CONVOLUTION(binary_mul);
  IPEX_OP_REGISTER_CONVOLUTION(binary_div);
  IPEX_OP_REGISTER_CONVOLUTION(binary_max);
  IPEX_OP_REGISTER_CONVOLUTION(binary_min);
  IPEX_OP_REGISTER_CONVOLUTION(binary_eq);
  IPEX_OP_REGISTER_CONVOLUTION(binary_ne);
  IPEX_OP_REGISTER_CONVOLUTION(binary_ge);
  IPEX_OP_REGISTER_CONVOLUTION(binary_gt);
  IPEX_OP_REGISTER_CONVOLUTION(binary_le);
  IPEX_OP_REGISTER_CONVOLUTION(binary_lt);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(sum);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_sub);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_mul);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_div);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_max);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_min);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_eq);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_ne);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_ge);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_gt);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_le);
  IPEX_OP_REGISTER_CONVOLUTION_SCALAR(binary_lt);
  IPEX_OP_REGISTER_CONVOLUTION(binary_mul_add);
  IPEX_OP_REGISTER_CONVOLUTION(sigmoid_binary_mul);
  IPEX_OP_REGISTER_CONVOLUTION(sigmoid_binary_mul_add);
  IPEX_OP_REGISTER_CONVOLUTION(sigmoid_binary_mul_add_relu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torch_ipex::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torch_ipex::_convolution_pointwise.binary(Tensor X, Tensor(a!) other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torch_ipex::_convolution_pointwise_.binary(Tensor(a!) other, Tensor X,Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor(a!) Y"));
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torch_ipex::_convolution_pointwise"),
      c10::DispatchKey::XPU,
      TORCH_FN(at::AtenIpexTypeXPU::convolution_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("torch_ipex::_convolution_pointwise.binary"),
      c10::DispatchKey::XPU,
      TORCH_FN(at::AtenIpexTypeXPU::convolution_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("torch_ipex::_convolution_pointwise_.binary"),
      c10::DispatchKey::XPU,
      TORCH_FN(at::AtenIpexTypeXPU::convolution_pointwise_binary_));
  m.impl(
      TORCH_SELECTIVE_NAME("torch_ipex::_convolution_pointwise"),
      c10::DispatchKey::Meta,
      TORCH_FN(at::AtenIpexTypeXPU::convolution_pointwise_meta));
}

} // namespace AtenIpexTypeXPU
} // namespace at
