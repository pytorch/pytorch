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
    num_first_symints(-1),
    num_first_tensors(-1),
    second(second_),
    num_second_symints(-1),
    num_second_tensors(-1) {}

std::vector<c10::SymInt> ChainedViewFunc::get_symints() {
  auto symints = first->get_symints();
  num_first_symints = symints.size();
  auto second_symints = second->get_symints();
  num_second_symints = second_symints.size();
  symints.reserve(symints.size() + num_second_symints);
  symints.insert(symints.end(), second_symints.begin(), second_symints.end());
  return symints;
}

void ChainedViewFunc::set_symints(const std::vector<c10::SymInt>& symints) {
  TORCH_INTERNAL_ASSERT(num_first_symints > -1);
  TORCH_INTERNAL_ASSERT(num_second_symints > -1);
  std::vector<c10::SymInt> first_symints(symints.begin(), symints.begin() + num_first_symints);
  first->set_symints(first_symints);
  std::vector<c10::SymInt> second_symints(symints.begin() + num_first_symints, symints.end());
  second->set_symints(second_symints);
}

std::vector<at::Tensor> ChainedViewFunc::get_tensors() {
  auto tensors = first->get_tensors();
  num_first_tensors = tensors.size();
  auto second_tensors = second->get_tensors();
  num_second_tensors = second_tensors.size();
  tensors.reserve(tensors.size() + num_second_tensors);
  tensors.insert(tensors.end(), second_tensors.begin(), second_tensors.end());
  return tensors;
}

void ChainedViewFunc::set_tensors(const std::vector<at::Tensor>& tensors) {
  TORCH_INTERNAL_ASSERT(num_first_tensors > -1);
  TORCH_INTERNAL_ASSERT(num_second_tensors > -1);
  std::vector<at::Tensor> first_tensors(tensors.begin(), tensors.begin() + num_first_tensors);
  first->set_tensors(first_tensors);
  std::vector<at::Tensor> second_tensors(tensors.begin() + num_first_tensors, tensors.end());
  second->set_tensors(second_tensors);
}

at::Tensor ChainedViewFunc::operator()(const at::Tensor& input_base) {
  return (*second)((*first)(input_base));
}

namespace generated {

${view_func_definitions}

} // namespace torch::autograd
} // namespace generated
