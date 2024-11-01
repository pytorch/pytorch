#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Array.h>

#include <atomic>
#include <utility>
#include <variant>

// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
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
//
// Supported device types for registration:
//   - CPU: Central Processing Unit
//   - CUDA: NVIDIA GPUs
//   - HIP: AMD GPUs
//   - MPS: Apple Silicon GPUs (Metal Performance Shaders)
//   - MTIA: Meta Training and Inference Devices
//   - XPU: Intel GPUs
//   - PrivateUse1: Reserved for private/custom device types
//
// If you want to update the list of supported devices, add a new dispatch_ptr
// member in DispatchStubImpl.h and update the get_call_ptr switch.
// As well you will need to update the inlined list in 'is_device_supported`
//
//
// ignore warnings about DispatchStub::DEFAULT, AVX, AVX2 defined elsewhere
C10_CLANG_DIAGNOSTIC_PUSH()
C10_CLANG_DIAGNOSTIC_IGNORE("-Wundefined-var-template")

namespace at::native {

enum class CPUCapability {
  DEFAULT = 0,
#if defined(HAVE_VSX_CPU_DEFINITION)
  VSX = 1,
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
  ZVECTOR = 1,
#elif defined(HAVE_SVE_CPU_DEFINITION)
  SVE256 = 1,
#else
  AVX2 = 1,
  AVX512 = 2,
#endif
  NUM_OPTIONS
};

// Enum for error types
enum class ErrorType {
  MissingDeviceKernel,
  DeviceNotSupported
};

// Alias for the return type using std::variant
using DispatchResult = std::variant<void*, ErrorType>;

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

  // The DispatchStubImpl::try_get_call_ptr() method is used to get the call
  // pointer for a given device type. If the call pointer is not found,
  // DispatchStubImpl::try_get_call_ptr() returns an ErrorType.
  // The main difference between try_get_call_ptr() and get_call_ptr() is that
  // try_get_call_ptr() will return the ErrorType and not raise an exception.
  DispatchResult try_get_call_ptr(
    c10::DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      , void *SVE256
#endif
  );

  // Analogous to try_get_call_ptr(), but it will return the ErrorType and not
  // raise an exception.
  DispatchResult try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
    , void *SVE256
#endif
  );


  void* get_call_ptr(
    c10::DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      , void *SVE256
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
    , void *SVE256
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
    void* cuda_dispatch_ptr;
    void* hip_dispatch_ptr;
    void* mps_dispatch_ptr;
    void* mtia_dispatch_ptr;
  #if defined(USE_XPU)
    void* xpu_dispatch_ptr;
  #endif
    void* privateuse1_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
    void* hip_dispatch_ptr = nullptr;
    void* mps_dispatch_ptr = nullptr;
    void* mtia_dispatch_ptr = nullptr;
  #if defined(USE_XPU)
    void* xpu_dispatch_ptr = nullptr;
  #endif
    void* privateuse1_dispatch_ptr = nullptr;
  #endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(const c10::DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      , reinterpret_cast<void*>(SVE256)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  #if defined(USE_XPU)
  void set_xpu_dispatch_ptr(FnPtr fn_ptr){
    impl.xpu_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }
  #endif

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

    void set_mtia_dispatch_ptr(FnPtr fn_ptr) {
    impl.mtia_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_privateuse1_dispatch_ptr(FnPtr fn_ptr) {
    impl.privateuse1_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  // Returns true if the dispatcher has a kernel registered for this device
  // type.
  bool is_device_supported(const c10::DeviceType device_type) {
    auto result = impl.try_get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      , reinterpret_cast<void*>(SVE256)
#endif
      );
    if (std::holds_alternative<ErrorType>(result)){
      return false;
    }
    return true;
  }

  static TORCH_API FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static TORCH_API FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static TORCH_API FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static TORCH_API FnPtr VSX;
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  static TORCH_API FnPtr ZVECTOR;
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  static TORCH_API FnPtr SVE256;
#endif
private:
  DispatchStubImpl impl;
};

namespace {
template <typename DispatchStub>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterXPUDispatch {
  RegisterXPUDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value){
    stub.set_xpu_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMPSDispatch {
  RegisterMPSDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_mps_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    // TODO: make this point at hip_dispatch_ptr
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMTIADispatch {
  RegisterMTIADispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_mtia_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterPRIVATEUSE1Dispatch {
  RegisterPRIVATEUSE1Dispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_privateuse1_dispatch_ptr(value);
  }
};

} // anonymous namespace
// Compiler will complain if you put things like std::tuple<Tensor, Tensor> in
// the `fn` argument of DECLARE_DISPATCH. Some possible workarounds, e.g.,
// adding parentheses and using helper struct to get rid of the parentheses, do
// not work with MSVC. So do a `using`-declaration if you need to pass in such
// `fn`, e.g., grid_sampler_2d_backward_cpu_kernel in GridSampleKernel.h.
#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    name##_DECLARE_DISPATCH_type() = default;                                              \
    name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;            \
    name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
    name##_DECLARE_DISPATCH_type(name##_DECLARE_DISPATCH_type&&) = delete;                 \
    name##_DECLARE_DISPATCH_type& operator=(name##_DECLARE_DISPATCH_type&&) = delete;      \
    ~name##_DECLARE_DISPATCH_type() = default;                                             \
  };                                                                                       \
  extern TORCH_API struct name##_DECLARE_DISPATCH_type name;

#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name##_DECLARE_DISPATCH_type::FnPtr TORCH_API DispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
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

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#define REGISTER_ZVECTOR_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, ZVECTOR, fn)
#else
#define REGISTER_ZVECTOR_DISPATCH(name, fn)
#endif

#ifdef HAVE_SVE256_CPU_DEFINITION
#define REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, SVE256, fn)
#else
#define REGISTER_SVE256_DISPATCH(name, fn)
#endif

// Macro to register the same kernel for all CPU arch types. This is useful
// if a kernel does not benefit from being recompiled across different arch types.
#define REGISTER_ALL_CPU_DISPATCH(name, fn)                                    \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)                                    \
  REGISTER_AVX512_DISPATCH(name, fn)                                           \
  REGISTER_AVX2_DISPATCH(name, fn)                                             \
  REGISTER_VSX_DISPATCH(name, fn)                                              \
  REGISTER_ZVECTOR_DISPATCH(name, fn)                                          \
  REGISTER_SVE256_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name)                                         \
  REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#define REGISTER_CUDA_DISPATCH(name, fn) \
  static RegisterCUDADispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

#define REGISTER_XPU_DISPATCH(name, fn) \
  static RegisterXPUDispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

#define REGISTER_HIP_DISPATCH(name, fn) \
  static RegisterHIPDispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

#define REGISTER_MPS_DISPATCH(name, fn) \
  static RegisterMPSDispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

#define REGISTER_MTIA_DISPATCH(name, fn) \
  static RegisterMTIADispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

#define REGISTER_PRIVATEUSE1_DISPATCH(name, fn) \
  static RegisterPRIVATEUSE1Dispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);

// NB: This macro must be used in an actual 'cu' file; if you try using
// it from a 'cpp' file it will not work!
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
// TODO: cut this over to HIP dispatch once we stop pretending that CUDA
// is HIP in the PyTorch HIPify build.
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// #define REGISTER_DISPATCH(name, fn) REGISTER_HIP_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
// NB: this macro must be used from a 'mm' file in order to dispatch a MPS kernel
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
// REGISTER_DISPATCH now dispatches an AVX512 kernel to nullptr but registers other dispatches.
// ALSO_REGISTER_AVX512_DISPATCH should be used for ensuring AVX512 dispatch, among others.
// ALSO_REGISTER_SVE256_DISPATCH should be used for ensuring SVE256 dispatch, among others.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, ((void*)(fn) ? nullptr : nullptr))
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define ALSO_REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
} // namespace at::native

C10_CLANG_DIAGNOSTIC_POP()
