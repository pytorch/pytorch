#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>

#include "torch/csrc/autograd/function.h"

// ${generated_comment}

// NOTE See [Sharded File] comment in VariableType

using namespace at;
using namespace torch::autograd::generated;
using torch::autograd::Node;

namespace torch {

namespace ProfiledType {

namespace {
${profiled_method_definitions}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  ${profiled_wrapper_registrations};
}

}  // namespace

} // namespace torch
