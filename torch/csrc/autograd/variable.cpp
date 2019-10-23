#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/Functions.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <typeinfo>

namespace torch {
namespace autograd {

AutogradMeta::~AutogradMeta() = default;

AutogradMeta::AutogradMeta(at::TensorImpl* self_impl, bool requires_grad, Edge gradient_edge) {
  grad_fn_ = std::move(gradient_edge.function);
  requires_grad_ = false;
  is_view_ = false;
  output_nr_ = gradient_edge.input_nr;

  // set_requires_grad also checks error conditions.
  set_requires_grad(requires_grad, self_impl);
  TORCH_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
}

DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base, Edge gradient_edge)
    : AutogradMeta(self_impl, false, std::move(gradient_edge)) {
  base_ = std::move(base);
  TORCH_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  self_impl->set_version_counter(base_.version_counter());
  attr_version = self_impl->version_counter().current_version();
}

DifferentiableViewMeta::~DifferentiableViewMeta() {
  base_.reset();
}

// In c++ file so we can access Node
void _create_cpp_hook(const at::Tensor& self) {
  auto &list = self.get_autograd_meta()->cpp_hooks_list;
  list.reset(new hooks_list());
  std::unique_ptr<FunctionPreHook> hook_ptr(new CppFunctionPreHook(list, self.output_nr()));
  self.clear_hooks();
  self.add_hook(std::make_shared<CppFunctionPreHook>(list, 0));
  auto fn = self.grad_fn();
  if (fn) {
    fn->add_pre_hook(std::move(hook_ptr));
  }
}

}} // namespace torch::autograd
