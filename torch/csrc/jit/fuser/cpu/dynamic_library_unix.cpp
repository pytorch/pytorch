#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/cpu/dynamic_library.h>
#include <torch/csrc/utils/disallow_copy.h>

#include <dlfcn.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

static void* checkDL(void* x) {
  if (!x) {
    AT_ERROR("error in dlopen or dlsym: ", dlerror());
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

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
