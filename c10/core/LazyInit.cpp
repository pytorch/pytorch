#include <c10/core/LazyInit.h>

namespace c10 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
C10_API lazy_init_ptr lazy_init_ptr_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];

void SetLazyInit(DeviceType t, lazy_init_ptr func) {
  lazy_init_ptr_array[static_cast<int>(t)] = func;
}

void LazyInit(const DeviceType& t) {
  lazy_init_ptr func = lazy_init_ptr_array[static_cast<int>(t)];
  if (func) {
      func();
  }
}

} // namespace c10
