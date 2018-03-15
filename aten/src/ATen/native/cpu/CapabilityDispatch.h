#include "ATen/cpu/cpuinfo/include/cpuinfo.h"
#include <type_traits>
#include <iostream>

namespace at {
namespace native {

enum class CPUCapability { DEFAULT, AVX, AVX2 };

#if defined(CPUCAPABILITYDEFAULT)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::DEFAULT;
#elif defined(CPUCAPABILITYAVX)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX;
#elif defined(CPUCAPABILITYAVX2)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX2;
#endif

template <typename FnType> struct DispatchStub {};

template <typename... ArgTypes>
struct DispatchStub<void(ArgTypes...)> {
  using FnType = void(ArgTypes...);

  template <template <CPUCapability> class allImpl,
            FnType **dispatch_ptr>
  static void init(ArgTypes... args) {
    cpuinfo_initialize();
    *dispatch_ptr = allImpl<CPUCapability::DEFAULT>::function;
    if (cpuinfo_has_x86_avx2()) {
      *dispatch_ptr = allImpl<CPUCapability::AVX2>::function;
    } else if (cpuinfo_has_x86_avx()) {
      *dispatch_ptr = allImpl<CPUCapability::AVX>::function;
    } 
    (*dispatch_ptr)(args...);
  }
};

}
}
