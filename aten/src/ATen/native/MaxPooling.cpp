#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/MaxPooling.h>
#include <ATen/native/Pool.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool1d_native.h>
#include <ATen/ops/max_pool1d_with_indices.h>
#include <ATen/ops/quantized_max_pool1d.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(max_pool1d_stub);

namespace {

static void check_max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 3,
      "max_pool1d() Expected 2D or 3D input tensor, but got ", self.sym_sizes());
  TORCH_CHECK(
      kernel_size.size() == 1,
      "max_pool1d() kernel_size must be an int, list of ints or tuple of ints of size 1 but got size ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 1,
      "max_pool1d() stride must be None, an int, list of ints, or tuple of ints of size 1 but got size ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1,
      "max_pool1d() padding must be an int, list of ints, or tuple of ints of size 1 but got size ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1,
      "max_pool1d() dilation must be an int, list of ints or tuple of ints of size 1 but got size ",
      dilation.size());

  // If stride=None then set it to kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  TORCH_CHECK(
      kernel_size[0] > 0,
      "max_pool1d() kernel_size must be greater than zero, but got ",
      kernel_size[0]);
  TORCH_CHECK(
      stride[0] > 0, "max_pool1d() stride must be greater than zero, but got ", stride[0]);
  TORCH_CHECK(
      padding[0] >= 0, "max_pool1d() padding must be non-negative, but got ", padding[0]);
  TORCH_CHECK(
      padding[0] <= kernel_size[0] / 2,
      "max_pool1d() padding should be at most half of kernel size, but got padding=",
      padding[0],
      " and kernel_size=",
      kernel_size[0]);
  TORCH_CHECK(
      dilation[0] > 0, "max_pool1d() dilation must be greater than zero, but got ", dilation[0]);

  const int64_t OW = pooling_output_shape(self.sym_size(-1).guard_int(__FILE__, __LINE__), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode);
  TORCH_CHECK(OW > 0, "max_pool1d() Invalid computed output size: ", OW);
}

} // namespace

namespace {

Tensor max_pool1d_impl(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;

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

  const int64_t OW = pooling_output_shape(IW, KW, PJ, SJ, DJ, ceil_mode);
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

  auto ndim = self.ndimension();
   TORCH_CHECK(
       (ndim == 2 && self.sym_size(0) != 0 && self.sym_size(1) != 0) ||
           (ndim == 3 && self.sym_size(1) != 0 && self.sym_size(2) != 0),
       "max_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input, but got:",
       self.sym_sizes());

  if (self.is_quantized()) {
    return at::quantized_max_pool1d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }

  check_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if ((self.requires_grad() && at::GradMode::is_enabled()) ||
      self._fw_grad(/*level */ 0).defined() ||
      !self.device().is_cpu() ||
      isTensorSubclassLike(self)) {
    // Needs indices for grad and with_indices defines CUDA dispatch
    return std::get<0>(at::max_pool1d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  return max_pool1d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace native
} // namespace at
