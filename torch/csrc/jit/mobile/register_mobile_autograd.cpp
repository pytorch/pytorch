#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <ATen/TypeDefault.h>

using Stack = std::vector<c10::IValue>;
using at::Tensor;
using at::Scalar;

namespace torch {
namespace autograd {
namespace VariableType {
Tensor mul_no_trace(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("mul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mul(self_, other_);
  })();
  auto result = std::move(tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}

Tensor add_no_trace(const Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("add", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::add(self_, other, alpha);
  })();
  auto result = std::move(tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
}
}
}

namespace {
static auto registry = torch::RegisterOperators().op(
    "_aten::add.Scalar",
    torch::RegisterOperators::options().kernel(c10::DispatchKey::VariableTensorId, &torch::autograd::VariableType::add_no_trace)
).op(
    "_aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
    torch::RegisterOperators::options().kernel(c10::DispatchKey::VariableTensorId, &torch::autograd::VariableType::mul_no_trace)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
);
}
