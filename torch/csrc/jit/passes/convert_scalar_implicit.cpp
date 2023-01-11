#include <torch/csrc/jit/passes/convert_scalar_implicit.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <ATen/core/type_factory.h>

namespace torch {
namespace jit {

void ConvertScalarImplicit(std::shared_ptr<Graph>& graph) {
  DepthFirstGraphNodeIterator it(graph);
  for (auto* node = it.next(); node != nullptr; node = it.next()) {
    if (node->kind() == c10::aten::ScalarImplicit) {
      Value * input = node->input(0);
      auto scalar_type = input->type()->cast<c10::TensorType>()->scalarType();
      TORCH_CHECK(scalar_type, "scalar type is not defined for input value");
      Value * output;
      if (c10::isIntegralType(*scalar_type, false)) {
        output = graph->insert(c10::aten::IntImplicit, {input});
      } else if (c10::isFloatingType(*scalar_type)) {
        output = graph->insert(c10::aten::FloatImplicit, {input});
      } else {
        throw std::runtime_error(
          "Expected isIntegralType or isFloatingType");
      }
      node->output()->replaceAllUsesWith(output);
      node->destroy();
    }
  }
}

} // namespace jit
} // namespace torch
