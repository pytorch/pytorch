#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Pooling.h>

namespace at {
namespace native {

DEFINE_DISPATCH(max_pool2d_stub);

namespace {

Tensor max_pool2d_impl(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;

  TORCH_CHECK(
      (input.dim() == 3 || input.dim() == 4),
      "max_pool2d: non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");

  PoolingParams p{};
  p.NB = input.dim() == 4 ? input.size(-4) : 1;
  p.NC = input.size(-3);
  p.IH = input.size(-2);
  p.IW = input.size(-1);
  p.KH = kernel_size[0];
  p.KW = kernel_size.size() == 1 ? p.KH : kernel_size[1];
  p.SI = stride.empty() ? p.KH : stride[0];
  p.SJ = stride.empty() ? p.KW : stride.size() == 1 ? p.SI : stride[1];
  p.PI = padding[0];
  p.PJ = padding.size() == 1 ? p.PI : padding[1];
  p.DI = dilation[0];
  p.DJ = dilation.size() == 1 ? p.DI : dilation[1];
  p.OH = pooling_output_shape<int64_t>(p.IH, p.KH, p.PI, p.SI, p.DI, ceil_mode);
  p.OW = pooling_output_shape<int64_t>(p.IW, p.KW, p.PJ, p.SJ, p.DJ, ceil_mode);

  pool2d_shape_check(
      input,
      p.KH,
      p.KW,
      p.SI,
      p.SJ,
      p.PI,
      p.PJ,
      p.DI,
      p.DJ,
      p.NC,
      p.IH,
      p.IW,
      p.OH,
      p.OW);

  Tensor output = at::empty({p.NB, p.NC, p.OH, p.OW}, input.options());
  max_pool2d_stub(input.device().type(), output, input, p);

  if (input.dim() == 3) {
    output.squeeze_(0);
  }

  guard.reset();
  namedinference::propagate_names(output, input);

  return output;
}

} // namespace

Tensor max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_max_pool2d(
          self, kernel_size, padding, stride, dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  if (self.requires_grad() || self.device() != at::kCPU) {
    return std::get<0>(at::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  // TODO (Heitor): Working on replacing with_indices version later
  // with new implementation.
  return max_pool2d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace native
} // namespace at