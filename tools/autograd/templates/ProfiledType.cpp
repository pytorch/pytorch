#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>

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

auto registerer = torch::import()
  ${profiled_wrapper_registrations};

}  // namespace

} // namespace torch
