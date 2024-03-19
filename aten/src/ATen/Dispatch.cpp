#include <ATen/Dispatch.h>
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
#include <ATen/record_function.h>

namespace at::detail {

void record_kernel_function_dtype(std::string name) {
  RECORD_FUNCTION_WITH_SCOPE(
        at::RecordScope::KERNEL_FUNCTION_DTYPE,
        std::move(name),
        c10::ArrayRef<const c10::IValue>{});
}

}  // namespace at::detail
#endif
