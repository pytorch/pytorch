#include <c10/util/Exception.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/Utils.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <Windows.h>
#endif

namespace at {


#ifndef _WIN32

// Unix

static void* checkDL(void* x) {
  if (!x) {
    AT_ERROR("Error in dlopen or dlsym: ", dlerror());
  }

  return x;
}
DynamicLibrary::DynamicLibrary(const char* name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  handle = checkDL(dlopen(name, RTLD_LOCAL | RTLD_NOW));
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  return checkDL(dlsym(handle, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle)
    return;
  dlclose(handle);
}

#else

// Windows

DynamicLibrary::DynamicLibrary(const char* name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  HMODULE theModule = LoadLibraryA(name);
  if (theModule) {
    handle = theModule;
  } else {
    AT_ERROR("error in LoadLibraryA");
  }
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  FARPROC procAddress = GetProcAddress((HMODULE)handle, name);
  if (!procAddress) {
    AT_ERROR("error in GetProcAddress");
  }
  return (void*)procAddress;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle) {
    return;
  }
  FreeLibrary((HMODULE)handle);
}

#endif

} // namespace at
