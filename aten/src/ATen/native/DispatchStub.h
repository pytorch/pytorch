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
struct DispatchStub;

/**
 * The sole purpose of this class is to outline methods that don't need to be
 * specialized or otherwise inlined and duplicated (by the compiler due to
 * template expansion), since it causes size bloat if there are a significant
 * number of specialization of the DispatchStub<> class.
 */
struct TORCH_API DispatchStubImpl {
  void* get_call_ptr(
    DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX_CPU_DEFINITION
      , void *AVX
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX_CPU_DEFINITION
    , void *AVX
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
    void* cuda_dispatch_ptr;
    void* hip_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
    void* hip_dispatch_ptr = nullptr;
  #endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

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
private:
  DispatchStubImpl impl;
};

namespace {
template <typename FnPtr, typename T>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename FnPtr, typename T>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
    // TODO: make this point at hip_dispatch_ptr
    stub.set_cuda_dispatch_ptr(value);
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
