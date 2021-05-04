#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>

namespace torch {
namespace jit {

uint64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kProducedBytecodeVersion;
}

} // namespace jit
} // namespace torch
