#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qglu_stub);
DEFINE_DISPATCH(qhardglu_stub);

Tensor& glu_quantized_cpu_out(Tensor& result, const Tensor& self, int64_t dim) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "qglu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);
  // size output to half of input
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[wrap_dim] = selfSize;
  result.resize_(newSizes);
  // half tensor
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);

  auto iter = TensorIterator::binary_op(result, firstHalf, secondHalf);
  qglu_stub(iter.device_type(), iter);
  return result;
}

Tensor glu_quantized_cpu(const Tensor& self, int64_t dim) {
  auto result = at::_empty_affine_quantized(
    {0},
    self.options(),
    self.q_scale(),
    self.q_zero_point());
  return glu_quantized_cpu_out(result, self, dim);
}

Tensor& hardglu_quantized_cpu_out(Tensor& result, const Tensor& self, int64_t dim) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "qglu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);
  // size output to half of input
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[wrap_dim] = selfSize;
  result.resize_(newSizes);
  // half tensor
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);

  auto iter = TensorIterator::binary_op(result, firstHalf, secondHalf);
  qhardglu_stub(iter.device_type(), iter);
  return result;
}

Tensor hardglu_quantized_cpu(const Tensor& self, int64_t dim) {
  auto result = at::_empty_affine_quantized(
    {0},
    self.options(),
    self.q_scale(),
    self.q_zero_point());
  return hardglu_quantized_cpu_out(result, self, dim);
}

}}  // namespace at::native
