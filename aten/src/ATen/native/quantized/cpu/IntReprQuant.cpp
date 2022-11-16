#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/int_repr_native.h>
#endif

namespace at {
namespace native {

// When input Tensor is non-dense, i.e. the allocated memory
// is larger than the memory used by all the elements, we'll
// convert it to dense tensor, otherwise we'll keep the memory
// format of the output the same as input
Tensor int_repr_quantized_cpu(const Tensor& self) {
  Tensor dst;
  // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self.scalar_type(), "int_repr", [&]() {
    if (bit_width == 4 || bit_width == 2) {
      int64_t out_size = at::ceil_div(self.numel() * bit_width, (int64_t)8);
      dst = at::empty(
          {out_size},
          self.options().dtype(UNDERLYING_TYPE),
          self.suggest_memory_format());
      const underlying_t* qdata = reinterpret_cast<underlying_t*>(self.data_ptr<scalar_t>());
      for (const auto i : c10::irange(dst.numel())) {
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
