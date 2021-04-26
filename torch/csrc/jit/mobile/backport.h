#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/backport.h>

namespace torch {
namespace jit {

int64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kProducedBytecodeVersion;
}

} // namespace jit
} // namespace torch
