#pragma once

#include "ATen/cpu/cpuinfo/include/cpuinfo.h"
#include <type_traits>
#include <iostream>

// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX) are
// compiled multiple times with different compiler flags (e.g. -mavx). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.
//
// Example:
//
// In native/cpu/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DispatchStub<fn_type> stub;
//
// In native/cpu/MyKernel.cpp:
//   void kernel(const Tensor& x) { ... }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(tensor);
//

namespace at {
namespace native {

enum class CPUCapability { DEFAULT, AVX, AVX2, NUM_OPTIONS };

template <typename FnPtr>
struct DispatchStub {
  static_assert(std::is_pointer<FnPtr>::value, "FnPtr should be a pointer type");

  template <typename... ArgTypes>
  void operator()(ArgTypes... args) {
    if (!dispatch_ptr) {
      dispatch_ptr = choose_impl();
    }
    (*dispatch_ptr)(args...);
  }

  FnPtr choose_impl() {
// Do not use cpuinfo on PowerPC as it shows confusing errors when run on ppc
#ifndef __powerpc__
    if (cpuinfo_initialize()) {
      int avx2 = static_cast<int>(CPUCapability::AVX2);
      if (!std::getenv("ATEN_DISABLE_AVX2") && cpuinfo_has_x86_avx2() && table[avx2]) {
        return table[avx2];
      }
      int avx = static_cast<int>(CPUCapability::AVX);
      if (!std::getenv("ATEN_DISABLE_AVX") && cpuinfo_has_x86_avx() && table[avx]) {
        return table[avx];
      }
    }
#endif
    int def = static_cast<int>(CPUCapability::DEFAULT);
    AT_ASSERT(table[def], "DispatchStub: missing default kernel");
    return table[def];
  }

  FnPtr dispatch_ptr = nullptr;
  FnPtr table[static_cast<int>(CPUCapability::NUM_OPTIONS)];
};


#if defined(CPU_CAPABILITY)

constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::CPU_CAPABILITY;

// Registers an implementation a kernel for the current CPU capability.
template<typename FnPtr>
struct RegisterDispatch {
  RegisterDispatch(DispatchStub<FnPtr>& stub, FnPtr value) {
    stub.table[static_cast<int>(CURRENT_CAPABILITY)] = value;
  }
};

// We only define the stub once in the DEFAULT capability compilation
#if defined(CPU_CAPABILITY_DEFAULT)
#define _DEFINE_STUB(stub, fn) DispatchStub<decltype(fn)> stub
#else
#define _DEFINE_STUB(stub, fn)
#endif

#define REGISTER_DISPATCH(stub, fn) \
  _DEFINE_STUB(stub, fn); \
  static RegisterDispatch<decltype(fn)> stub ## __register(stub, fn);

#endif

}
}
