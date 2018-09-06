#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER
#pragma once

#include "dlfcn.h"

namespace torch { namespace jit { namespace cpufuser {

static void* checkDL(void* x) {
  if (!x) {
    AT_ERROR("error in dlopen or dlsym: ", dlerror());
  }

  return x;
}

struct DynamicLibrary {
  TH_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  DynamicLibrary(const char* name) {
    handle = checkDL(dlopen(name, RTLD_LOCAL | RTLD_NOW));
  }

  void* sym(const char* name) {
    JIT_ASSERT(handle);
    return checkDL(dlsym(handle, name));
  }

  ~DynamicLibrary() {
    if (!handle) return;
    dlclose(handle);
  }

private:
  void* handle = nullptr;
};

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // USE_CPU_FUSER
