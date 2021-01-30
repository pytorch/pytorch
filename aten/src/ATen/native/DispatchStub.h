#pragma once

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <type_traits>
#include <atomic>

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
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub);
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
//
// TODO: CPU instruction set selection should be folded into whatever
// the main dispatch mechanism is.

// ignore warnings about DispatchStub::DEFAULT, AVX, AVX2 defined elsewhere
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#endif

namespace at { namespace native {

enum class CPUCapability {
  DEFAULT = 0,
#ifdef HAVE_VSX_CPU_DEFINITION
  VSX = 1,
#else
  AVX = 1,
  AVX2 = 2,
#endif
  NUM_OPTIONS
};

CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct TORCH_API DispatchStub;

template <typename rT, typename T, typename... Args>
struct TORCH_API DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    FnPtr call_ptr = nullptr;
    if (device_type == DeviceType::CPU) {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        fptr = choose_cpu_impl();
        cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
      }
      call_ptr = fptr;
    } else if (device_type == DeviceType::CUDA) {
      AT_ASSERTM(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
      call_ptr = cuda_dispatch_ptr;
    } else if (device_type == DeviceType::HIP) {
      AT_ASSERTM(hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
      call_ptr = hip_dispatch_ptr;
    }
    if (call_ptr == nullptr) {
      AT_ERROR("DispatchStub: unsupported device type", device_type);
    }
    return call_ptr;
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  FnPtr choose_cpu_impl() {
    auto capability = static_cast<int>(get_cpu_capability());
    (void)capability;
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (capability >= static_cast<int>(CPUCapability::AVX2)) {
      AT_ASSERTM(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    }
#endif
#ifdef HAVE_AVX_CPU_DEFINITION
    if (capability >= static_cast<int>(CPUCapability::AVX)) {
      AT_ASSERTM(AVX, "DispatchStub: missing AVX kernel");
      return AVX;
    }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    if (capability >= static_cast<int>(CPUCapability::VSX)) {
      AT_ASSERTM(VSX, "DispatchStub: missing VSX kernel");
      return VSX;
    }
#endif
    AT_ASSERTM(DEFAULT, "DispatchStub: missing default kernel");
    return DEFAULT;
  }

// Fixing dispatch error in Windows debug builds.
// See https://github.com/pytorch/pytorch/issues/22681 for more details.
#if defined(_MSC_VER) && defined(_DEBUG)
  std::atomic<FnPtr> cpu_dispatch_ptr;
  FnPtr cuda_dispatch_ptr;
  FnPtr hip_dispatch_ptr;
#else
  std::atomic<FnPtr> cpu_dispatch_ptr{nullptr};
  FnPtr cuda_dispatch_ptr = nullptr;
  FnPtr hip_dispatch_ptr = nullptr;
#endif
  static FnPtr DEFAULT;
#ifdef HAVE_AVX_CPU_DEFINITION
  static FnPtr AVX;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static FnPtr VSX;
#endif
};

namespace {
template <typename FnPtr, typename T>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
    stub.cuda_dispatch_ptr = value;
  }
};

template <typename FnPtr, typename T>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
    // TODO: make this point at hip_dispatch_ptr
    stub.cuda_dispatch_ptr = value;
  }
};
} // anonymous namespace

// Compiler will complain if you put things like std::tuple<Tensor, Tensor> in
// the `fn` argument of DECLARE_DISPATCH. Some possible workarounds, e.g.,
// adding parentheses and using helper struct to get rid of the parentheses, do
// not work with MSVC. So do a `using`-declaration if you need to pass in such
// `fn`, e.g., grid_sampler_2d_backward_cpu_kernel in GridSampleKernel.h.
#define DECLARE_DISPATCH(fn, name)         \
  struct name : DispatchStub<fn, name> {   \
    name() = default;                      \
    name(const name&) = delete;            \
    name& operator=(const name&) = delete; \
  };                                       \
  extern TORCH_API struct name name

#define DEFINE_DISPATCH(name) struct name name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;

#ifdef HAVE_AVX_CPU_DEFINITION
#define REGISTER_AVX_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX, fn)
#else
#define REGISTER_AVX_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

#ifdef HAVE_VSX_CPU_DEFINITION
#define REGISTER_VSX_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, VSX, fn)
#else
#define REGISTER_VSX_DISPATCH(name, fn)
#endif

#define REGISTER_NO_CPU_DISPATCH(name, fn_type)                                \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, static_cast<fn_type>(nullptr))         \
  REGISTER_AVX_DISPATCH(name, static_cast<fn_type>(nullptr))                   \
  REGISTER_AVX2_DISPATCH(name, static_cast<fn_type>(nullptr))          \
  REGISTER_VSX_DISPATCH(name, static_cast<fn_type>(nullptr))

#define REGISTER_CUDA_DISPATCH(name, fn) \
  static RegisterCUDADispatch<decltype(fn), struct name> name ## __register(name, fn);

#define REGISTER_HIP_DISPATCH(name, fn) \
  static RegisterHIPDispatch<decltype(fn), struct name> name ## __register(name, fn);

// NB: This macro must be used in an actual 'cu' file; if you try using
// it from a 'cpp' file it will not work!
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
// TODO: cut this over to HIP dispatch once we stop pretending that CUDA
// is HIP in the PyTorch HIPify build.
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// #define REGISTER_DISPATCH(name, fn) REGISTER_HIP_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif


}} // namespace at::native


#if defined(__clang__)
#pragma clang diagnostic pop
#endif
