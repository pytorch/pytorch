#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {

// When input Tensor is non-dense, i.e. the allocated memory
// is larger than the memory used by all the elements, we'll
// convert it to dense tensor, otherwise we'll keep the memory
// format of the output the same as input
Tensor int_repr_quantized_cpu(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self.scalar_type(), "int_repr", [&]() {
    if (bit_width == 4) {
      int64_t out_size = std::ceil(self.numel() * 0.5);
      dst = at::empty(
          {out_size},
          self.options().dtype(UNDERLYING_TYPE),
          self.suggest_memory_format());
      const underlying_t* qdata = reinterpret_cast<underlying_t*>(self.data_ptr<scalar_t>());
      for (int64_t i = 0; i < dst.numel(); ++i) {
        dst[i] = static_cast<underlying_t>(qdata[i]);
      }
    } else {
      dst = at::empty(
          self.sizes(),
          self.options().dtype(UNDERLYING_TYPE),
          self.suggest_memory_format());
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(dst)
        .add_input(self)
        .build();
      cpu_kernel(iter, [](scalar_t value) -> underlying_t { return value.val_; });
      }
  });
  return dst;
}

} // namespace native
} // namespace at
