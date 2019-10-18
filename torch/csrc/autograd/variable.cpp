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
// NB: output_nr is input_nr on an Edge!
AutogradMeta::AutogradMeta(at::TensorImpl* self_impl, bool requires_grad, std::shared_ptr<Node> grad_fn, uint32_t output_nr) {
  grad_fn_ = std::move(grad_fn);
  requires_grad_ = false;
  is_view_ = false;
  output_nr_ = output_nr;

  // set_requires_grad also checks error conditions.
  set_requires_grad(requires_grad, self_impl);
  TORCH_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
}

DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base, std::shared_ptr<Node> grad_fn, uint32_t output_nr)
    : AutogradMeta(self_impl, false, std::move(grad_fn), output_nr) {
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

}} // namespace torch::autograd
