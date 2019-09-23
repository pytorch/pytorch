#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/GatedLinearUnit.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(glu_stub);

Tensor& glu_out(Tensor &result, const Tensor& self, int64_t dim) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              dim, " is size ", nIn);

  // size output to half of input
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[dim] = selfSize;
  result.resize_(newSizes);

  // halve tensor
  Tensor firstHalf = self.narrow(dim, 0, selfSize);
  Tensor secondHalf = self.narrow(dim, selfSize, selfSize);

  auto iter = TensorIterator::binary_op(result, firstHalf, secondHalf);
  glu_stub(iter.device_type(), iter);
  return result;
}

Tensor glu(const Tensor& self, int64_t dim) {
  auto result = at::empty({0}, self.options());
  return at::glu_out(result, self, dim);
}

} // at::native
} // at
