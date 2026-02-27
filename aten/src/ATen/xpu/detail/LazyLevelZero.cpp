#include <ATen/xpu/detail/LazyLevelZero.h>

#include <ATen/DynamicLibrary.h>
#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <stdexcept>

namespace at::xpu::detail {
namespace _stubs {

at::DynamicLibrary& getZELibrary() {
#if defined(_WIN32)
  static at::DynamicLibrary lib("ze_loader.dll");
#else
  static at::DynamicLibrary lib("libze_loader.so");
#endif
  return lib;
}

#define _STUB_1(LIB, NAME, RETTYPE, ARG1)                                     \
  RETTYPE NAME(ARG1 a1) {                                                     \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(a1);                                                            \
  }

#define _STUB_2(LIB, NAME, RETTYPE, ARG1, ARG2)                               \
  RETTYPE NAME(ARG1 a1, ARG2 a2) {                                            \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(a1, a2);                                                        \
  }

#define _STUB_3(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3)                         \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                   \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(a1, a2, a3);                                                    \
  }

#define _STUB_4(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4)                   \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                          \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(a1, a2, a3, a4);                                                \
  }

#define _STUB_5(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4, ARG5)             \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4, ARG5 a5) {                 \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(a1, a2, a3, a4, a5);                                            \
  }

#define ZE_STUB1(NAME, A1) _STUB_1(ZE, NAME, ze_result_t ZE_APICALL, A1)
#define ZE_STUB2(NAME, A1, A2) _STUB_2(ZE, NAME, ze_result_t ZE_APICALL, A1, A2)
#define ZE_STUB3(NAME, A1, A2, A3) \
  _STUB_3(ZE, NAME, ze_result_t ZE_APICALL, A1, A2, A3)
#define ZE_STUB4(NAME, A1, A2, A3, A4) \
  _STUB_4(ZE, NAME, ze_result_t ZE_APICALL, A1, A2, A3, A4)
#define ZE_STUB5(NAME, A1, A2, A3, A4, A5) \
  _STUB_5(ZE, NAME, ze_result_t ZE_APICALL, A1, A2, A3, A4, A5)

// Intel level zero is not defaultly available on Windows.
#ifndef _WIN32
ZE_STUB5(
    zeModuleCreate,
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_module_desc_t*,
    ze_module_handle_t*,
    ze_module_build_log_handle_t*)
ZE_STUB3(
    zeKernelCreate,
    ze_module_handle_t,
    const ze_kernel_desc_t*,
    ze_kernel_handle_t*)
ZE_STUB2(zeKernelGetProperties, ze_kernel_handle_t, ze_kernel_properties_t*)
ZE_STUB4(
    zeMemGetAllocProperties,
    ze_context_handle_t,
    const void*,
    ze_memory_allocation_properties_t*,
    ze_device_handle_t*)
ZE_STUB3(
    zeModuleBuildLogGetString,
    ze_module_build_log_handle_t,
    size_t*,
    char*)
ZE_STUB1(zeModuleBuildLogDestroy, ze_module_build_log_handle_t)

#endif

} // namespace _stubs

LevelZero lazyLevelZero = {
// Intel level zero is not defaultly available on Windows.
#ifndef _WIN32
#define _REFERENCE_MEMBER(name) _stubs::name,
    AT_FORALL_ZE(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
#endif
};
} // namespace at::xpu::detail
