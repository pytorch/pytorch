#pragma once

#include "ATen/cpu/cpuinfo/include/cpuinfo.h"
#include <string>
#include <type_traits>

namespace at {
namespace native {

enum class CPUCapability { DEFAULT, AVX, AVX2 };

#if defined(CPU_CAPABILITY_DEFAULT)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::DEFAULT;
#elif defined(CPU_CAPABILITY_AVX)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX;
#elif defined(CPU_CAPABILITY_AVX2)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX2;
#endif

static std::string CPUCapability_to_string(CPUCapability c) {
  if (c == CPUCapability::DEFAULT)
    return "DEFAULT";
  if (c == CPUCapability::AVX)
    return "AVX";
  if (c == CPUCapability::AVX2)
    return "AVX2";
  return "UNDEFINED!";
}

template <typename FnType> struct DispatchStub {};

template <typename... ArgTypes> struct DispatchStub<void(ArgTypes...)> {
  using FnType = void(ArgTypes...);

  template <template <CPUCapability> class allImpl, FnType **dispatch_ptr>
  static void init(ArgTypes... args) {
    cpuinfo_initialize();
    if (cpuinfo_has_x86_avx2()) {
      *dispatch_ptr = allImpl<CPUCapability::AVX2>::function;
    } else if (cpuinfo_has_x86_avx()) {
      *dispatch_ptr = allImpl<CPUCapability::AVX>::function;
    } else {
      *dispatch_ptr = allImpl<CPUCapability::DEFAULT>::function;
    }
    (*dispatch_ptr)(args...);
  }
};
}
}
