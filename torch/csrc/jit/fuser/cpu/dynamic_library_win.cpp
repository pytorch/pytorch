#include <Windows.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/cpu/dynamic_library.h>
#include <torch/csrc/utils/disallow_copy.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {


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

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
