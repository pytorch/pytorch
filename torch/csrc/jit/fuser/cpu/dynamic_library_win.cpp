#include <torch/csrc/jit/assertions.h>
#include <torch/csrc/jit/fuser/cpu/dynamic_library.h>
#include <torch/csrc/utils/disallow_copy.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

DynamicLibrary::DynamicLibrary(const char* name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  AT_ERROR("NYI: DynamicLibrary on Windows");
}

void* DynamicLibrary::sym(const char* name) {
  AT_ERROR("NYI: DynamicLibrary on Windows");
}

DynamicLibrary::~DynamicLibrary() {}

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
