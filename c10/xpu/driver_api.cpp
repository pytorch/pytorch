#include <c10/xpu/driver_api.h>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#include <fmt/os.h>
#else
#include <dlfcn.h>
#endif

namespace c10::xpu {

namespace {

#ifdef _WIN32
void* ze_dlopen(const char* name) {
  return LoadLibraryA(name);
}

void* ze_dlsym(void* handle, const char* name) {
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(handle), name));
}
#else
void* ze_dlopen(const char* name) {
  return dlopen(name, RTLD_LAZY);
}

void* ze_dlsym(void* handle, const char* name) {
  return dlsym(handle, name);
}
#endif // _WIN32

void* get_ze_symbol(const char* name) {
#ifdef _WIN32
  static void* handle = ze_dlopen("ze_loader.dll");
#else
  static void* handle = ze_dlopen("libze_loader.so");
#endif
  TORCH_CHECK(handle, "Can't load level zero loader library");
  return ze_dlsym(handle, name);
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

// Generates a stub for NAME that DriverAPI::NAME##_ initially points to. On
// first call, it resolves the real symbol, patches DriverAPI::NAME##_ so
// later calls skip the stub, then forwards this call to it.
#define ZE_STUB(NAME, ...)                                                \
  ze_result_t ZE_APICALL NAME(_STUB_PARAMS(__VA_ARGS__)) {                \
    auto fn = reinterpret_cast<decltype(&NAME)>(get_ze_symbol(__func__)); \
    TORCH_CHECK(fn, "Can't get symbol " C10_STRINGIZE(NAME));             \
    DriverAPI::get()->NAME##_ = fn;                                       \
    return fn(_STUB_ARGS(__VA_ARGS__));                                   \
  }

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

#undef ZE_STUB
#undef _STUB_ARGS
#undef _STUB_PARAMS
#undef _STUB_DISPATCH
#undef _STUB_NARGS
#undef _STUB_NARGS_IMPL
#undef _STUB_ARGS_5
#undef _STUB_ARGS_4
#undef _STUB_ARGS_3
#undef _STUB_ARGS_2
#undef _STUB_ARGS_1
#undef _STUB_PARAMS_5
#undef _STUB_PARAMS_4
#undef _STUB_PARAMS_3
#undef _STUB_PARAMS_2
#undef _STUB_PARAMS_1

} // namespace

DriverAPI create_driver_api() {
  DriverAPI r{};
#define CREATE_MEMBER(name) r.name##_ = name;
  C10_LIBXPU_DRIVER_API_REQUIRED(CREATE_MEMBER)
#undef CREATE_MEMBER
  return r;
}

DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace c10::xpu
