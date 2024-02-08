#include "torch/csrc/autograd/generated/ViewFuncs.h"

// ${generated_comment}

using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd {

std::vector<c10::SymInt> ChainedViewFunc::get_symints() const {
  auto symints = first->get_symints();
  auto second_symints = second->get_symints();
  symints.reserve(symints.size() + second_symints.size());
  symints.insert(
      symints.end(),
      std::make_move_iterator(second_symints.begin()),
      std::make_move_iterator(second_symints.end()));
  return symints;
}

std::vector<at::Tensor> ChainedViewFunc::get_tensors() const {
  auto tensors = first->get_tensors();
  auto second_tensors = second->get_tensors();
  tensors.reserve(tensors.size() + second_tensors.size());
  tensors.insert(
      tensors.end(),
      std::make_move_iterator(second_tensors.begin()),
      std::make_move_iterator(second_tensors.end()));
  return tensors;
}

at::Tensor ChainedViewFunc::operator()(const at::Tensor& input_base) const {
  return (*second)((*first)(input_base));
}

std::unique_ptr<ViewFunc> ChainedViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  std::optional<std::vector<c10::SymInt>> first_symints;
  std::optional<std::vector<c10::SymInt>> second_symints;
  if (symints.has_value()) {
    TORCH_INTERNAL_ASSERT(symints->size() == num_symints());
    first_symints = std::vector<c10::SymInt>(
        symints->begin(), symints->begin() + first->num_symints());
    second_symints = std::vector<c10::SymInt>(
        symints->begin() + first->num_symints(), symints->end());
  }

  std::optional<std::vector<at::Tensor>> first_tensors;
  std::optional<std::vector<at::Tensor>> second_tensors;
  if (tensors.has_value()) {
    TORCH_INTERNAL_ASSERT(tensors->size() == num_tensors());
    first_tensors = std::vector<at::Tensor>(
        tensors->begin(), tensors->begin() + first->num_tensors());
    second_tensors = std::vector<at::Tensor>(
        tensors->begin() + first->num_tensors(), tensors->end());
  }

  return std::make_unique<ChainedViewFunc>(
      first->clone_and_set(first_symints, first_tensors),
      second->clone_and_set(second_symints, second_tensors));
}

namespace generated {

${view_func_definitions}

} // namespace torch::autograd
} // namespace generated
