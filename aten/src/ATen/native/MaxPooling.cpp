#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/MaxPooling.h>

namespace at {
namespace native {

DEFINE_DISPATCH(max_pool1d_stub);

namespace {

// Compute the output size for the given pooling parameters
inline int64_t output_size(
    int64_t input_size,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {
  int64_t num = input_size + 2 * padding - dilation * (kernel_size - 1) - 1;
  // Ensure last kernel window starts within bounds in ceil mode
  if (ceil_mode && stride - dilation * (kernel_size - 1) <= num % stride) {
    return (num + stride - 1) / stride + 1;
  }
  return num / stride + 1;
}

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

  const int64_t OW = output_size(IW, KW, SJ, PJ, DJ, ceil_mode);
  TORCH_CHECK(OW >= 0, "max_pool1d() Invalid computed output size: ", OW);
  Tensor output = at::empty({NB, NC, OW}, self.options());

  PoolingParams params{NB, NC, IW, OW, KW, SJ, PJ, DJ};
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
  if (self.requires_grad() || !self.device().is_cpu()) {
    return std::get<0>(at::max_pool1d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  return max_pool1d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace native
} // namespace at
