#ifndef C10_MOBILE

#include <c10/cuda/CUDADriverAPI.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <c10/util/win32-headers.h>
#endif

namespace c10 {
namespace cuda {

#ifndef _WIN32

void CUDADriverAPI::initialize_api() {
#if defined(__APPLE__)
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
  std::string libcuda = "libcuda.dylib";
#else // if Linux
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
  std::string libcuda = "libcuda.so.1";
#endif
  handle = dlopen(libcaffe2_nvrtc.c_str(), RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    handle = dlopen(libcuda.c_str(), RTLD_LOCAL | RTLD_NOW);
  }
  if (!handle) {
    TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
  } else {
    _hasPrimaryContext_funcptr = (_cuDevicePrimaryCtxGetState)dlsym(
        handle, "cuDevicePrimaryCtxGetState");
    if (!_hasPrimaryContext_funcptr) {
      TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
    }
#if defined(USE_CUDA)
    _getLastErrorString_funcptr =
        (_cuGetErrorString)dlsym(handle, "cuGetErrorString");
    if (!_getLastErrorString_funcptr) {
      TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
    }
#endif
  }
}

void CUDADriverAPI::destroy_handle() {
  if (!handle) {
    return;
  }
  dlclose(handle);
}

#else // if _WIN32

void CUDADriverAPI::initialize_api() {
  LPCWSTR caffe2_nvrtc = L"caffe2_nvrtc.dll";
  HMODULE theModule;
  bool reload = true;
  // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
  if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") !=
      NULL) {
    theModule =
        LoadLibraryExW(caffe2_nvrtc, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
      reload = false;
    }
  }

  if (reload) {
    theModule = LoadLibraryW(caffe2_nvrtc);
  }

  if (theModule) {
    handle = (void*)theModule;
  } else {
    char buf[256];
    DWORD dw = GetLastError();
    FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buf,
        (sizeof(buf) / sizeof(char)),
        NULL);
    TORCH_WARN_ONCE(
        false,
        " WinError in LoadLibrary for caffe2_nvrtc.dll. ",
        dw,
        ": ",
        buf);
  }

  FARPROC procAddress =
      GetProcAddress((HMODULE)handle, "cuDevicePrimaryCtxGetState");
  if (!procAddress) {
    TORCH_WARN_ONCE(
        false,
        " error in GetProcAddress. CUDA Driver API: cuDevicePrimaryCtxGetState is not available.");
  } else {
    _hasPrimaryContext_funcptr = (_cuDevicePrimaryCtxGetState)procAddress;
#if defined(USE_CUDA)
    FARPROC procAddress2 = GetProcAddress((HMODULE)handle, "cuGetErrorString");
    if (!procAddress2) {
      TORCH_WARN_ONCE(
          false,
          " error in GetProcAddress. CUDA Driver API: cuGetErrorString is not available.");
    } else {
      _getLastErrorString_funcptr = (_cuGetErrorString)procAddress2;
    }
#endif
  }
}

void CUDADriverAPI::destroy_handle() {
  if (!handle) {
    return;
  }
  FreeLibrary((HMODULE)handle);
}

#endif // _WIN32

CUDADriverAPI::CUDADriverAPI() {
  is_api_initialized = false;
  handle = nullptr;
  _hasPrimaryContext_funcptr = nullptr;
  _getLastErrorString_funcptr = nullptr;
}

CUDADriverAPI::~CUDADriverAPI() {
  destroy_handle();
}

bool CUDADriverAPI::hasPrimaryContext(int device) {
  if (!is_api_initialized) {
    initialize_api();
    is_api_initialized = true;
  }
  if (!_hasPrimaryContext_funcptr) {
    return true;
  }

  int active = 0;
  unsigned int flags = 0;
  CUresult err = _hasPrimaryContext_funcptr(device, &flags, &active);
  if (err != CUDA_SUCCESS) {
    RAISE_WARNING(err);
    return true;
  }

  return active == 1;
}

void CUDADriverAPI::RAISE_WARNING(CUresult err) {
  if (!_getLastErrorString_funcptr) {
    TORCH_WARN("CUDA driver error: ", static_cast<int>(err));
  } else {
    const char* err_str;
    CUresult get_error_str_err = _getLastErrorString_funcptr(err, &err_str);
    if (get_error_str_err != CUDA_SUCCESS) {
      TORCH_WARN("CUDA driver error: unknown error");
    } else {
      TORCH_WARN("CUDA driver error: ", err_str);
    }
  }
}

} // namespace cuda
} // namespace c10

#endif // C10_MOBILE
