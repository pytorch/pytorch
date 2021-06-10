#include <ATen/ATen.h>

namespace at {

namespace {
Tensor trivial_empty() {
  return detail::empty_generic(
    /*size=*/{1},
    /*allocator=*/getCPUAllocator(),
    /*dispatch_key=*/DispatchKey::CPU,
    /*scalar_type=*/ScalarType::Float,
    /*device=*/c10::Device(c10::kCPU),
    /*memory_format_opt=*/c10::MemoryFormat::Contiguous
  );
}
}  // namespace

namespace native {

Tensor _noop_unary(const Tensor& self) {
  return trivial_empty();
}

Tensor _noop_unary_manual(const Tensor& self) {
  return trivial_empty();
}

Tensor _noop_binary(const Tensor& self, const Tensor& other) {
  return trivial_empty();
}

Tensor _noop_binary_manual(const Tensor& self, const Tensor& other) {
  return trivial_empty();
}

}  // namespace native
}  // namespace at
