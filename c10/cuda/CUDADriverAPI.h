#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <c10/util/win32-headers.h>
#endif

#include <c10/util/Exception.h>

#ifndef C10_MOBILE
#ifndef _WIN32
class CUDADriverAPI {
 public:
  CUDADriverAPI() {
    std::cout << "CUDADriverAPI created\n";

    std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
    handle = dlopen(libcaffe2_nvrtc.c_str(), RTLD_LOCAL | RTLD_NOW);
    if (!handle) {
      TORCH_CHECK(false, "Error in dlopen: ", dlerror());
    }

    _c10_hasPrimaryContext = (_cuDevicePrimaryCtxGetState)dlsym(
        handle, "cuDevicePrimaryCtxGetState");
    if (!_c10_hasPrimaryContext) {
      TORCH_CHECK(false, "Error in dlopen: ", dlerror());
    }
  }

  bool c10_hasPrimaryContext(int device) {
    int active = 0;
    unsigned int flags = 0;
    CUresult err = _c10_hasPrimaryContext(device, &flags, &active);

    TORCH_WARN("CUDA driver error: ", static_cast<int>(err));

    return active == 1;
  }

  ~CUDADriverAPI() {
    destroy_handle();
    std::cout << "CUDADriverAPI destroyed\n";
  }

 private:
  void* handle = nullptr;
  typedef CUresult (*_cuDevicePrimaryCtxGetState)(
      CUdevice dev,
      unsigned int* flags,
      int* active);
  _cuDevicePrimaryCtxGetState _c10_hasPrimaryContext = nullptr;

  void destroy_handle() {
    dlclose(handle);
  }
};
#else // if _WIN32
class CUDADriverAPI {
 public:
  CUDADriverAPI() {
    std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dll";

    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    HMODULE theModule;
    bool reload = true;
    auto wname = c10::u8u16(libcaffe2_nvrtc);
    // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
    if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") !=
        NULL) {
      theModule =
          LoadLibraryExW(wname.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
        reload = false;
      }
    }

    if (reload) {
      theModule = LoadLibraryW(wname.c_str());
    }

    if (theModule) {
      handle = theModule;
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
      TORCH_CHECK(
          false,
          "error in LoadLibrary for ",
          libcaffe2_nvrtc,
          ". WinError ",
          dw,
          ": ",
          buf);
    }

    FARPROC procAddress =
        GetProcAddress((HMODULE)handle, "cuDevicePrimaryCtxGetState");
    if (!procAddress) {
      TORCH_CHECK(false, "error in GetProcAddress");
    }

    _c10_hasPrimaryContext = (_cuDevicePrimaryCtxGetState)procAddress;
  }

  bool c10_hasPrimaryContext(int device) {
    int active = 0;
    unsigned int flags = 0;
    CUresult err = _c10_hasPrimaryContext(device, &flags, &active);

    TORCH_WARN("CUDA driver error: ", static_cast<int>(err));

    return active == 1;
  }

  ~CUDADriverAPI() {
    destroy_handle();
    std::cout << "CUDADriverAPI destroyed\n";
  }

 private:
  void* handle = nullptr;
  typedef CUresult (*_cuDevicePrimaryCtxGetState)(
      CUdevice dev,
      unsigned int* flags,
      int* active);
  _cuDevicePrimaryCtxGetState _c10_hasPrimaryContext = nullptr;

  void destroy_handle() {
    if (!handle || leak_handle) {
      return;
    }
    FreeLibrary((HMODULE)handle);
  }
};
#endif // _WIN32
#endif // C10_MOBILE
