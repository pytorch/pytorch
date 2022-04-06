#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_guard.h>

namespace torch {
namespace jit {

bool is_upgraders_enabled() {
  return ENABLE_UPGRADERS;
}

} // namespace jit
} // namespace torch
