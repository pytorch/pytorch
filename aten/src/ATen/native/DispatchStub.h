#pragma once

#include <ATen/Error.h>
#include <ATen/ScalarType.h>
#include <type_traits>

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
//   stub(kCPU, tensor);
//

namespace at {
namespace native {

enum class CPUCapability {
  DEFAULT = 0,
  AVX = 1,
  AVX2 = 2,
  NUM_OPTIONS
};

CPUCapability get_cpu_capability();

template <typename FnPtr>
struct DispatchStub {
  static_assert(std::is_pointer<FnPtr>::value, "FnPtr should be a pointer type");

  template <typename... ArgTypes>
  void operator()(Backend backend, ArgTypes... args) {
    if (backend == Backend::CPU) {
      if (!dispatch_ptr) {
        dispatch_ptr = choose_cpu_impl();
      }
      (*dispatch_ptr)(args...);
    } else if (backend == Backend::CUDA) {
      AT_ASSERTM(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
      (*cuda_dispatch_ptr)(args...);
    } else {
      AT_ERROR("DispatchStub: unsupported backend", backend);
    }
  }

  FnPtr choose_cpu_impl() {
    int def = static_cast<int>(CPUCapability::DEFAULT);
    int avx = static_cast<int>(CPUCapability::AVX);
    int avx2 = static_cast<int>(CPUCapability::AVX2);

    auto capability = static_cast<int>(get_cpu_capability());
    if (capability >= avx2 && table[avx2]) {
      return table[avx2];
    }
    if (capability >= avx && table[avx]) {
      return table[avx];
    }
    AT_ASSERTM(table[def], "DispatchStub: missing default kernel");
    return table[def];
  }

  FnPtr dispatch_ptr = nullptr;
  FnPtr cuda_dispatch_ptr = nullptr;
  FnPtr table[static_cast<int>(CPUCapability::NUM_OPTIONS)];
};


#if defined(CPU_CAPABILITY) || defined(__CUDACC__)

namespace {

template <typename FnPtr>
struct RegisterDispatch {
  RegisterDispatch(DispatchStub<FnPtr>& stub, FnPtr value) {
#if defined(__CUDACC__)
    stub.cuda_dispatch_ptr = value;
#else
    int cap = static_cast<int>(CPUCapability::CPU_CAPABILITY);
    AT_ASSERT(!stub.table[cap])
    stub.table[cap] = value;
#endif
  }
};

} // anonymous namespace

#define REGISTER_DISPATCH(stub, fn) \
  static RegisterDispatch<decltype(fn)> stub ## __register(stub, fn);

#endif

}
}
