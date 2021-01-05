#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

Tensor int_repr_quantized_cuda(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr_quantized_cuda", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(dst)
      .add_input(self)
      .build();
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t value) -> underlying_t {
      return value.val_;
    });
  });
  return dst;
}

} // namespace native
} // namespace at
