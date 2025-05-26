#include "torch/csrc/autograd/FunctionsManual.h"
#include "torch/csrc/dynamo/compiled_autograd.h"

// ${generated_comment}

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace torch::autograd::generated::details;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd::generated {

static at::IValue compute_output_metadata(const torch::autograd::edge_list& next_edges) {
  auto output_metadata = torch::dynamo::autograd::IValuePacker<
      std::vector<std::optional<InputMetadata>>>::pack(
              torch::dynamo::autograd::get_input_metadata(next_edges));
  return output_metadata;
}

static C10_NOINLINE variable_list compiled_autograd_apply_functional(
    const PackedArgs& packed_args,
    const edge_list& next_edges,
    SwapSavedVariables& saved,
    const variable_list& grads,
    const std::string& name) {
  auto output_metadata = compute_output_metadata(next_edges);
  const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
  return pyinterface->call_function(
      saved.get_py_compiler(),
      "apply_functional",
      name,
      grads,
      packed_args.vec(),
      output_metadata);
}

${autograd_function_definitions}

} // namespace torch::autograd::generated
