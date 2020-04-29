#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

Tensor int_repr_quant_cuda(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr_quant_cuda", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIterator();
    iter.add_output(dst);
    iter.add_input(self);
    iter.dont_compute_common_dtype();
    iter.build();
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t value) -> underlying_t {
      return value.val_;
    });
  });
  return dst;
}

} // namespace native
} // namespace at
