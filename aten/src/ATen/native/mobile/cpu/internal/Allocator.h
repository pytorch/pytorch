#pragma once

#include <ATen/native/mobile/cpu/internal/Common.h>

#ifdef USE_XNNPACK

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {

class Allocator final : public at::Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() override = default;

  static void deleter(void* buffer);

  virtual DataPtr allocate(size_t nbytes) const override;
  virtual DeleterFnPtr raw_deleter() const override;

 private:
  static constexpr uint32_t kGuard = XNN_EXTRA_BYTES;
};

at::Tensor new_tensor(
    IntArrayRef size,
    const TensorOptions& options,
    c10::MemoryFormat memory_format);

} // namespace internal
} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
