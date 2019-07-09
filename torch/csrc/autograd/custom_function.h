#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace torch { namespace autograd {

std::vector<Variable> _wrap_outputs(std::unordered_set<Variable*> &inputs,
  std::vector<Variable*> &raw_outputs, std::unordered_set<Variable*> &non_differentiable,
  std::vector<Variable*> &dirty_inputs, std::shared_ptr<Function> cdata);

}} // namespace torch::autograd
