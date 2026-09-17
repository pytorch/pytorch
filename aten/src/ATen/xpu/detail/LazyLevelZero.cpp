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

// Generates the typed parameter list / forwarding argument list for a stub
// of a given arity, e.g. _STUB_PARAMS_2(int, float) -> "int a1, float a2".
#define _STUB_PARAMS_1(A1) A1 a1
#define _STUB_PARAMS_2(A1, A2) A1 a1, A2 a2
#define _STUB_PARAMS_3(A1, A2, A3) A1 a1, A2 a2, A3 a3
#define _STUB_PARAMS_4(A1, A2, A3, A4) A1 a1, A2 a2, A3 a3, A4 a4
#define _STUB_PARAMS_5(A1, A2, A3, A4, A5) A1 a1, A2 a2, A3 a3, A4 a4, A5 a5

// The types aren't needed to forward the call, only their count.
#define _STUB_ARGS_1(...) a1
#define _STUB_ARGS_2(...) a1, a2
#define _STUB_ARGS_3(...) a1, a2, a3
#define _STUB_ARGS_4(...) a1, a2, a3, a4
#define _STUB_ARGS_5(...) a1, a2, a3, a4, a5

// Counts the number of variadic ARGn types (1-5) passed to ZE_STUB
#define _STUB_NARGS_IMPL(_1, _2, _3, _4, _5, N, ...) N
#define _STUB_NARGS(...) \
  C10_EXPAND_MSVC_WORKAROUND(_STUB_NARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1))

#define _STUB_DISPATCH(HELPER, NARG, ...) \
  C10_CONCATENATE(HELPER, NARG)(__VA_ARGS__)
#define _STUB_PARAMS(...) \
  _STUB_DISPATCH(_STUB_PARAMS_, _STUB_NARGS(__VA_ARGS__), __VA_ARGS__)
#define _STUB_ARGS(...) \
  _STUB_DISPATCH(_STUB_ARGS_, _STUB_NARGS(__VA_ARGS__), __VA_ARGS__)

#define _STUB(LIB, NAME, RETTYPE, ...)                                        \
  RETTYPE NAME(_STUB_PARAMS(__VA_ARGS__)) {                                   \
    auto fn =                                                                 \
        reinterpret_cast<decltype(&NAME)>(get##LIB##Library().sym(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));                 \
    lazyLevelZero.NAME = fn;                                                  \
    return fn(_STUB_ARGS(__VA_ARGS__));                                       \
  }

#define ZE_STUB(NAME, ...) _STUB(ZE, NAME, ze_result_t ZE_APICALL, __VA_ARGS__)

// Intel level zero is not defaultly available on Windows.
#ifndef _WIN32
ZE_STUB(
    zeModuleCreate,
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_module_desc_t*,
    ze_module_handle_t*,
    ze_module_build_log_handle_t*)
ZE_STUB(
    zeKernelCreate,
    ze_module_handle_t,
    const ze_kernel_desc_t*,
    ze_kernel_handle_t*)
ZE_STUB(zeKernelGetProperties, ze_kernel_handle_t, ze_kernel_properties_t*)
ZE_STUB(
    zeMemGetAllocProperties,
    ze_context_handle_t,
    const void*,
    ze_memory_allocation_properties_t*,
    ze_device_handle_t*)
ZE_STUB(zeModuleBuildLogGetString, ze_module_build_log_handle_t, size_t*, char*)
ZE_STUB(zeModuleBuildLogDestroy, ze_module_build_log_handle_t)

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
