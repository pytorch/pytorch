#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/MaxPooling.h>
#include <ATen/native/Pool.h>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(max_pool1d_stub);

namespace {

Tensor max_pool1d_impl(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;

  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 3,
      "max_pool1d() input tensor must have 2 or 3 dimensions but got ",
      self.dim());
  TORCH_CHECK(
      kernel_size.size() == 1,
      "max_pool1d() kernel_size must be an int or int list of size 1 but got size ",
      kernel_size.size());
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1,
      "max_pool1d() stride must be None, an int or int list of size 1 but got size ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1,
      "max_pool1d() padding must be an int or int list of size 1 but got size ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1,
      "max_pool1d() dilation must be an int or int list of size 1 but got size ",
      dilation.size());

  // If stride=None then set it to kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  const int64_t NB = self.dim() == 3 ? self.size(-3) : 1;
  const int64_t NC = self.size(-2);
  const int64_t IW = self.size(-1);
  const int64_t KW = kernel_size[0];
  const int64_t SJ = stride[0];
  const int64_t PJ = padding[0];
  const int64_t DJ = dilation[0];

  TORCH_CHECK(
      KW > 0,
      "max_pool1d() kernel_size must be greater than zero, but got ",
      KW);
  TORCH_CHECK(
      SJ > 0, "max_pool1d() stride must be greater than zero, but got ", SJ);
  TORCH_CHECK(
      PJ >= 0, "max_pool1d() padding must be non-negative, but got ", PJ);
  TORCH_CHECK(
      PJ <= KW / 2,
      "max_pool1d() padding should be at most half of kernel size, but got padding=",
      PJ,
      " and kernel_size=",
      KW);
  TORCH_CHECK(
      DJ > 0, "max_pool1d() dilation must be greater than zero, but got ", DJ);

  const int64_t OW = pooling_output_shape(IW, KW, PJ, SJ, DJ, ceil_mode);
  TORCH_CHECK(OW >= 0, "max_pool1d() Invalid computed output size: ", OW);
  Tensor output = at::empty({NB, NC, OW}, self.options());

  PoolingParams1D params{NB, NC, IW, OW, KW, SJ, PJ, DJ};
  max_pool1d_stub(self.device().type(), output, self, params);

  if (self.dim() == 2) {
    output.squeeze_(0);
  }

  guard.reset();
  namedinference::propagate_names(output, self);

  return output;
}

} // namespace

Tensor max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool1d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  if ((self.requires_grad() && at::GradMode::is_enabled()) ||
      !self.device().is_cpu()) {
    // Needs indices for grad and with_indices defines CUDA dispatch
    return std::get<0>(at::max_pool1d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  return max_pool1d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace native
} // namespace at
