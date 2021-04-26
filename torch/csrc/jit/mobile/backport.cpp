#include <caffe2/serialize/versions.h>

namespace torch {
namespace jit {

int64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kProducedBytecodeVersion;
}

} // namespace jit
} // namespace torch
