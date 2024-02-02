#include "torch/csrc/autograd/generated/ViewFuncs.h"

// ${generated_comment}

using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd {

ChainedViewFunc::ChainedViewFunc(
  const std::shared_ptr<ViewFunc>& first_,
  const std::shared_ptr<ViewFunc>& second_)
  : first(first_),
    num_first_symints(0),
    num_first_tensors(0),
    second(second_),
    num_second_symints(0),
    num_second_tensors(0) {}

std::vector<c10::SymInt> ChainedViewFunc::get_symints() const {
  auto symints = first->get_symints();
  num_first_symints = symints.size();
  auto second_symints = second->get_symints();
  num_second_symints = second_symints.size();
  symints.reserve(symints.size() + num_second_symints);
  symints.insert(symints.end(), second_symints.begin(), second_symints.end());
  return symints;
}

std::vector<at::Tensor> ChainedViewFunc::get_tensors() const {
  auto tensors = first->get_tensors();
  num_first_tensors = tensors.size();
  auto second_tensors = second->get_tensors();
  num_second_tensors = second_tensors.size();
  tensors.reserve(tensors.size() + num_second_tensors);
  tensors.insert(tensors.end(), second_tensors.begin(), second_tensors.end());
  return tensors;
}

at::Tensor ChainedViewFunc::operator()(
    const at::Tensor& input_base,
    const std::optional<std::vector<c10::SymInt>>& symints,
    const std::optional<std::vector<at::Tensor>>& tensors) const {
  std::optional<std::vector<c10::SymInt>> first_symints;
  std::optional<std::vector<c10::SymInt>> second_symints;
  if (symints.has_value()) {
    TORCH_INTERNAL_ASSERT(symints->size() == num_first_symints + num_second_symints);
    first_symints = std::vector<c10::SymInt>(symints->begin(), symints->begin() + num_first_symints);
    second_symints = std::vector<c10::SymInt>(symints->begin() + num_first_symints, symints->end());
  }

  std::optional<std::vector<at::Tensor>> first_tensors;
  std::optional<std::vector<at::Tensor>> second_tensors;
  if (tensors.has_value()) {
    TORCH_INTERNAL_ASSERT(tensors->size() == num_first_tensors + num_second_tensors);
    first_tensors = std::vector<at::Tensor>(tensors->begin(), tensors->begin() + num_first_tensors);
    second_tensors = std::vector<at::Tensor>(tensors->begin() + num_first_tensors, tensors->end());
  }

  // NB: guarding is done below
  auto first_output = (*first)(input_base, first_symints, first_tensors);
  return (*second)(first_output, second_symints, second_tensors);
}

namespace generated {

${view_func_definitions}

} // namespace torch::autograd
} // namespace generated
