#pragma once

#include <c10/core/Device.h>

namespace c10 {

using lazy_init_ptr = void (*)();
/** Set the lazy init func for DeviceType `t`.Note that this is not
 *  not thread-safe, and we assume this function will only be called
 *  during initialization.
 */
C10_API void SetLazyInit(DeviceType t, lazy_init_ptr func);
C10_API void LazyInit(const DeviceType& t);

template <DeviceType t>
struct LazyInitRegisterer {
  explicit LazyInitRegisterer(lazy_init_ptr func) {
    SetLazyInit(t, func);
  }
};

#define REGISTER_LAZY_INIT(t, f)                      \
  namespace {                                         \
  static c10::LazyInitRegisterer<t> g_lazy_init_d(f); \
  }
} // namespace c10
