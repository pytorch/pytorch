#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

namespace torch { namespace autograd {

variable_list _wrap_outputs(const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Function> &cdata) {

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
      var.clear_hooks();
      if (auto grad_acc_fn = var.try_get_grad_accumulator()) {
        auto grad_acc = dynamic_cast<AccumulateGrad*>(grad_acc_fn.get());
        grad_acc->variable.reset();
      }
      if (cdata) {
        var.rebase_history({cdata, output_nr});
      }
    } else if (is_input) {
      // An input has been returned, but it wasn't modified. Return it as a view
      // so that we can attach a new grad_fn to the Variable.
      var = var.view_as(var);
      var.set_gradient_edge({cdata, output_nr});
    } else if (cdata) {
      var.set_gradient_edge({cdata, output_nr});
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


template <typename T, typename... Args>
void extract_vars(variable_list& list, T&& cur, Args&& ... args) {
  extract_vars(list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(variable_list& list, Variable&& cur, Args&& ... args) {
  list.push_back(cur);
  extract_vars(list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(variable_list& list, Variable& cur, Args&& ... args) {
  list.push_back(cur);
  extract_vars(list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(variable_list& list, Args&& ... args) {
}


template<class T>
template<typename... Args>
variable_list CFunction<T>::apply(Args&&... args) {
  variable_list input_vars;
  extract_vars(input_vars, std::forward<Args>(args)...);

  bool is_executable =  GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  auto next_edges = collect_next_edges(input_vars);

  std::shared_ptr<CustomFunc<T>> grad_fn(new CustomFunc<T>, deleteFunction);
  grad_fn->ctx.cdata = grad_fn;
  grad_fn->set_next_edges(std::move(next_edges));
  grad_fn->clear_input_metadata();


  variable_list outputs;
  {
    AutoGradMode grad_mode(false);
    outputs = T::forward(&grad_fn->ctx, std::forward<Args>(args)...);
  }

  return _wrap_outputs(input_vars, grad_fn->ctx.get_non_differentiable(), grad_fn->ctx.get_dirty(), outputs, is_executable ? grad_fn : nullptr);
}

template<class T>
variable_list CustomFunc<T>::apply(variable_list&& inputs) {
  auto outputs = T::backward(&ctx, inputs);

  auto num_forward_inputs = inputs.size();
  auto num_outputs = outputs.size();

  // Returning too many results is ok, but only as long as they're all undefined.
  // Truncate the result vector in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (int i = num_forward_inputs; i < num_outputs; ++i) {
      all_undef &= (outputs[i].defined());
    }
    if (all_undef) {
      outputs.resize(num_forward_inputs);
      num_outputs = num_forward_inputs;
    }
  }

  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got " ;
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  return outputs;
}

template<class T>
void CustomFunc<T>::release_variables() {
  ctx.clear_saved();
}

void AutogradContext::save_for_backward(const variable_list &to_save) {
  saved_variables.clear();
  saved_variables.reserve(to_save.size());
  for(auto& var : to_save) {
    saved_variables.emplace_back(var, (var.grad_fn().get() == grad_fn.get()));
  }
}

void AutogradContext::mark_dirty(const variable_list &inputs) {
  dirty_inputs.clear();
  dirty_inputs.reserve(inputs.size());
  for(auto& var : inputs) {
    dirty_inputs.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::mark_non_differentiable(const variable_list &outputs) {
  non_differentiable.clear();
  non_differentiable.reserve(outputs.size());
  for(auto& var : outputs) {
    non_differentiable.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::clear_saved() {
  saved_variables.clear();
}

variable_list AutogradContext::get_saved_variables() const {
  variable_list saved;
  saved.reserve(saved_variables.size());
  for (auto& var : saved_variables) {
    saved.push_back(var.unpack(grad_fn));
  }
  return saved;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_dirty() const {
  return dirty_inputs;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_non_differentiable() const {
  return non_differentiable;
}

}} // namespace torch::autograd
