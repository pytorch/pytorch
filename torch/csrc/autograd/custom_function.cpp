#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

namespace torch { namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
  : layout(var.layout())
  , device(var.device())
  , scalar_type(var.scalar_type())
  , size(var.sizes().vec())
  , requires_grad(var.requires_grad()) {
}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  return at::zeros(size,
    at::TensorOptions(scalar_type).device(device).layout(layout));
}

variable_list _wrap_outputs(const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Node> &cdata) {

  std::unordered_set<at::TensorImpl*> inputs;
  inputs.reserve(input_vars.size());
  for (auto& var : input_vars) {
    inputs.emplace(var.unsafeGetTensorImpl());
  }

  // Sets the grad_fn and output_nr of an output Variable.
  auto set_history = [&](Variable& var, uint32_t output_nr, bool is_input, bool is_modified,
                         bool is_differentiable) {
    if (!is_differentiable) {
      if (!var.requires_grad()) {
        return;
      }
      // NB: we don't support returning non-differentiable views that could require grad
      if (var.is_view()) {
        throw std::runtime_error("Returning Variables sharing storage with other Variables "
                                 "that require grad is not supported in Python functions. "
                                 "Please submit a feature request if you hit this error.");
      }
      // Return detached aliases of inputs, instead of changing their requires_grad
      // property.
      if (is_input) {
        var = var.detach();
      } else {
        var.detach_();
      }
    } else if (is_modified) {
      if (var.is_leaf() && var.requires_grad()) {
        throw std::runtime_error("a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // If the input was modified, transplant the grad_fn in the graph:
      // grad_fn <- variable <- self  ==>  grad_fn <- self <- variable
      var.grad().reset();
      impl::clear_hooks(var);
      if (auto grad_acc_fn = impl::try_get_grad_accumulator(var)) {
        auto grad_acc = dynamic_cast<AccumulateGrad*>(grad_acc_fn.get());
        grad_acc->variable.reset();
      }
      if (cdata) {
        impl::rebase_history(var, {cdata, output_nr});
      }
    } else if (is_input) {
      // An input has been returned, but it wasn't modified. Return it as a view
      // so that we can attach a new grad_fn to the Variable.
      var = var.view_as(var);
      impl::set_gradient_edge(var, {cdata, output_nr});
    } else if (cdata) {
      impl::set_gradient_edge(var, {cdata, output_nr});
    }
  };

  int num_outputs = raw_outputs.size();

  std::vector<torch::autograd::Variable> outputs;
  outputs.reserve(num_outputs);

  for (auto i = 0; i < num_outputs; ++i) {
    auto out_tensor_impl = raw_outputs[i].unsafeGetTensorImpl();
    bool is_input = inputs.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    bool is_differentiable = cdata && non_differentiable.count(out_tensor_impl) == 0;

    Variable var = raw_outputs[i];

    if (cdata) {
      auto output_nr = cdata->add_input_metadata(var);
      AT_ASSERT(i == (int)output_nr);
    }
    set_history(var, i, is_input, is_modified, is_differentiable);

    outputs.emplace_back(var);
  }

  return outputs;
}

void check_variable_result(const Variable& original, const Variable& result, std::string hook_name) {
  if (original.type() != result.type()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value (";
    ss << "was " << original.toString() << " got ";
    ss << result.toString() << ")";
    throw std::runtime_error(ss.str());
  }

  if (original.is_cuda() != result.is_cuda()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value";
    if (original.is_cuda()) {
      ss << " (was CUDA tensor got CPU tensor)";
    } else {
      ss << " (was CPU tensor got CUDA tensor)";
    }
    throw std::runtime_error(ss.str());
  }

  if (original.sizes().vec() != result.sizes().vec()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the size of value";
    throw std::runtime_error(ss.str());
  }
}

void AutogradContext::save_for_backward(variable_list to_save) {
  to_save_ = std::move(to_save);
}

// The logic for handling saved variables here is the same as python_function.cpp
// See _save_variables() and unpack_saved_variables()
void AutogradContext::save_variables() {
  saved_variables_.clear();
  auto ptr = grad_fn_.lock();

  for (const auto& var : to_save_) {
    // Allow empty variables to be saved
    if (var.defined()) {
      bool is_output = var.grad_fn().get() == ptr.get();
      saved_variables_.emplace_back(var, is_output);
    } else {
      saved_variables_.emplace_back();
    }
  }
  to_save_.clear();
}

variable_list AutogradContext::get_saved_variables() const {
  TORCH_CHECK(!has_freed_buffers_, ERR_BACKWARD_TWICE);
  variable_list saved;
  saved.reserve(saved_variables_.size());
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  for (auto& var : saved_variables_) {
    saved.push_back(var.unpack(ptr));
  }
  return saved;
}

void AutogradContext::mark_dirty(const variable_list &inputs) {
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  for(auto& var : inputs) {
    dirty_inputs_.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::mark_non_differentiable(const variable_list &outputs) {
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  for(auto& var : outputs) {
    non_differentiable_.insert(var.unsafeGetTensorImpl());
  }
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_dirty() const {
  return dirty_inputs_;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_non_differentiable() const {
  return non_differentiable_;
}
}} // namespace torch::autograd
