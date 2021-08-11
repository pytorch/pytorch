#include <c10/util/irange.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/variable.h>

namespace torch { namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
  : layout(var.layout())
  , device(var.device())
  , scalar_type(var.scalar_type())
  , size(var.sizes().vec())
  , requires_grad(var.requires_grad())
  , is_empty(false) {
}

VariableInfo::VariableInfo() : requires_grad(false), is_empty(true) {}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  if (is_empty) {
    // Return undefined tensor.
    return at::Tensor();
  } else {
    return at::zeros(
        size, at::TensorOptions(scalar_type).device(device).layout(layout));
  }
}

template <typename F>
void _foreach_tensor(
    F fn,
    torch::jit::Stack* stack,
    size_t stack_start,
    size_t size) {
  // Enumerate over tensors in a stack, including ones in TensorLists
  int idx = 0;
  for (const auto i : c10::irange(size)) {
    auto& ivalue = (*stack)[stack_start + i];
    if (ivalue.isTensor()) {  // true for optional tensor that has value
      const auto& tensor = ivalue.toTensor();
      fn(idx, i, tensor);
      idx++;
    } else if (ivalue.isTensorList()) {
      for (const auto& iv : ivalue.toListRef()) {
        const auto& tensor = iv.toTensor();
        fn(idx, i, tensor);
        idx++;
      }
    }
  }
}

void autogradNotImplementedFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // Mimics the logic of a VariableType NotImplemented kernel
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto& arguments = schema.arguments();
  const auto& returns = schema.returns();
  const auto num_arguments = arguments.size();
  const auto num_returns = returns.size();
  const auto stack_start = stack->size() - num_arguments;
  const bool grad_mode = GradMode::is_enabled();
  std::vector<const at::Tensor*> tensors_requiring_grad;

  // Keep track of which outputs are output of in-place modification
  // so we can rebase_history if necessary
  std::vector<bool> is_inplace_output;
  std::vector<bool> is_aliased_output;
  is_inplace_output.reserve(num_returns);
  is_aliased_output.reserve(num_returns);
  for (const auto i : c10::irange(num_returns)) {
    const auto& alias_info = returns[i].alias_info();
    is_inplace_output.push_back(alias_info.has_value() && alias_info->isWrite());
    is_aliased_output.push_back(alias_info.has_value());
  }
  size_t num_tensor_inputs = 0;  // Only used for DEBUG-only checks

  _foreach_tensor([&](size_t _, size_t idx_arg, const at::Tensor& t) {
    TORCH_CHECK(t.defined(), "Expected argument ", idx_arg, " of ", op_name, " to be defined.");
    if (grad_mode && t.requires_grad()) {
      tensors_requiring_grad.push_back(&t);
    }
    num_tensor_inputs++;
    TORCH_CHECK_NOT_IMPLEMENTED(!generated::details::isFwGradDefined(t), "Trying to use forward AD with ", op_name, " that does not support it.");
  }, stack, stack_start, num_arguments);

  const bool any_requires_grad = tensors_requiring_grad.size() > 0;

  _foreach_tensor([&](size_t _, size_t i, const at::Tensor& t) {
    const auto& alias_info = arguments[i].alias_info();
    if (alias_info.has_value() && alias_info->isWrite()) {
      check_inplace(t, any_requires_grad);
    }
  }, stack, stack_start, num_arguments);

  std::shared_ptr<NotImplemented> grad_fn;
  if (any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented(op_name), deleteNode);
    grad_fn->set_next_edges(collect_next_edges(tensors_requiring_grad));
  }

  #ifndef NDEBUG
  // See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
  std::vector<c10::IValue> stack_copy(*stack);
  std::vector<c10::intrusive_ptr<TensorImpl>> impl_saved;
  impl_saved.reserve(num_tensor_inputs);
  std::vector<c10::optional<Storage>> storage_saved;
  storage_saved.reserve(num_tensor_inputs);
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    storage_saved.push_back(t.has_storage() ? c10::optional<Storage>(t.storage()) : c10::nullopt);
  }, &stack_copy, stack_start, num_arguments);
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    impl_saved.push_back(t.getIntrusivePtr());
  }, &stack_copy, stack_start, num_arguments);
  #endif
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
  }
  #ifndef NDEBUG
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    if (storage_saved.at(idx).has_value())
      AT_ASSERT(storage_saved.at(idx).value().is_alias_of(t.storage()));
  }, &stack_copy, stack_start, num_arguments);
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    if (impl_saved.at(idx))
      AT_ASSERT(impl_saved.at(idx) == t.getIntrusivePtr());
  }, &stack_copy, stack_start, num_arguments);
  // Do we have alias information for tensors in tensorlist outputs?
  _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
    if (!is_inplace_output[idx_ret])
      AT_ASSERT(t.use_count() <= 1);  // Okay to return undefined tensor
  }, stack, stack->size() - num_returns, num_returns);
  // TODO: Add check if view ops return tensor that is aliased with the right input
  _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
    if (!is_aliased_output[idx_ret] && t.has_storage())
      AT_ASSERT(t.storage().use_count() == 1);
  }, stack, stack->size() - num_returns, num_returns);
  #endif
  const auto& ret = torch::jit::last(stack, num_returns);

  if (any_requires_grad) {
    for (const auto return_idx : c10::irange(ret.size())) {
      if (ret[return_idx].isTensor()) {
        const auto& t = ret[return_idx].toTensor();
        if (isDifferentiableType(t.scalar_type())) {
          if (is_inplace_output[return_idx]) {
            rebase_history(const_cast<at::Tensor&>(t), grad_fn);
          } else {
            set_history(const_cast<at::Tensor&>(t), grad_fn);
          }
        }
      }
    }
  }
}

std::vector<c10::optional<Variable>> _wrap_outputs(const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<c10::optional<Variable>> raw_outputs,
  const std::shared_ptr<Node> &cdata) {

  std::unordered_set<at::TensorImpl*> inputs;
  inputs.reserve(input_vars.size());
  for (auto& var : input_vars) {
    inputs.emplace(var.unsafeGetTensorImpl());
  }

  int num_outputs = raw_outputs.size();

  // Sets the grad_fn and output_nr of an output Variable.
  auto set_history = [&](Variable& var, uint32_t output_nr, bool is_input, bool is_modified,
                         bool is_differentiable) {
    if (!is_differentiable) {
      if (!var.requires_grad()) {
        return;
      }
      // Return detached aliases of inputs, instead of changing their requires_grad
      // property.
      if (is_input) {
        var = var.detach();
      } else if (!var.is_view()) {
        var.detach_();
      }
      // If var is a view of one of the inputs of the custom autograd Function,
      // we don't detach it in a no_grad block. This is so that we can mimic the
      // behavior of returning a view from a no_grad block:
      //   x = torch.randn(3, requires_grad=True)
      //   with torch.no_grad():
      //       y = x.view(-1)
      // Here, `y` requires_grad (!).
    } else if (is_modified) {
      if (var.is_leaf() && var.requires_grad()) {
        throw std::runtime_error("a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // No need to mark as modified Tensors that are not inputs.
      if (!is_input) {
        TORCH_WARN("Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
                   " is no need to pass it to mark_dirty().");
      }
      // If the input is a view, the rebase will need to rewrite the graph and this only works if we have a single
      // output to this Function.
      TORCH_CHECK(!(var.is_view() && num_outputs > 1), "If your Function modifies inplace an input that is a view"
                  " of another Tensor, your Function cannot return more than one Tensor. This is not supported"
                  " by the current autograd engine. You should either make sure the input is not a view (using"
                  " .clone() for example) or make your Function only return one Tensor (potentially splitting"
                  " it into two Functions: one doing the inplace that returns a single Tensor and a second one"
                  " that does the other operations). You can ask on the forum https://discuss.pytorch.org/ if"
                  " you need help to do this change.");

      // If the input was modified, transplant the grad_fn in the graph:
      // grad_fn <- variable <- self  ==>  grad_fn <- self <- variable
      var.mutable_grad().reset();
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
      // Run in no_grad mode to mimic the behavior of the forward.
      {
        AutoGradMode grad_mode(false);
        var = var.view_as(var);
      }
      impl::set_gradient_edge(var, {cdata, output_nr});
    } else if (cdata) {
      impl::set_gradient_edge(var, {cdata, output_nr});
    }
  };

  std::vector<c10::optional<Variable>> outputs;
  std::unordered_set<at::TensorImpl*> outputs_impl; // For dirty_inputs check
  outputs.reserve(num_outputs);
  int num_diff_outputs = 0;


  for (const auto i : c10::irange(num_outputs)) {
    // For outputs that are not tensors, put a placeholder undefined input.
    if (!raw_outputs[i].has_value()) {
      if (cdata) {
        auto output_nr = cdata->add_input_metadata(Node::undefined_input());
        AT_ASSERT(i == (int)output_nr);
      }
      outputs.emplace_back();
      continue;
    }

    Variable var = raw_outputs[i].value();

    auto out_tensor_impl = var.unsafeGetTensorImpl();
    bool is_input = inputs.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    bool is_differentiable = cdata && non_differentiable.count(out_tensor_impl) == 0
                              && isDifferentiableType(var.scalar_type());

    if (cdata) {
      auto output_nr = cdata->add_input_metadata(var);
      AT_ASSERT(i == (int)output_nr);
    }
    set_history(var, i, is_input, is_modified, is_differentiable);

    // For deprecation cycle. Can be removed after 1.6. In the case where we detected a view
    // in no grad mode during the forward, only warn the user (do not change the flag if we
    // return and input that is a view as is).
    // See NOTE [ View + Inplace detection ] for why we replace everything by a warning.
    if (!(is_input && is_modified) && var.is_view()) {
      // is_view() => diff_view_meta
      auto diff_view_meta = impl::get_view_autograd_meta(var);
      diff_view_meta->set_creation_meta(CreationMeta::IN_CUSTOM_FUNCTION);
    }

    if (is_differentiable) {
      ++num_diff_outputs;
    }

    outputs_impl.insert(out_tensor_impl);
    outputs.emplace_back(var);
  }

  // If multiple differentiable outputs are returned, we do not allow views to be modified inplace
  // See NOTE [ View + Inplace detection ] for more details
  if (num_diff_outputs > 1) {
    for (auto& var: outputs) {
      if (var.has_value()) {
        auto diff_view_meta = impl::get_view_autograd_meta(var.value());
        if (diff_view_meta && diff_view_meta->has_bw_view()) {
          diff_view_meta->set_creation_meta(CreationMeta::MULTI_OUTPUT_NODE);
        }
      }
    }
  }

  // All the modified Tensors must be returned as is for the rewrite to be valid.
  for (auto& dirty_input : dirty_inputs) {
    TORCH_CHECK(outputs_impl.count(dirty_input) > 0,
                "Some elements marked as dirty during the forward method were not returned as output. The"
                " inputs that are modified inplace must all be outputs of the Function.");
  }

  return outputs;
}

void check_variable_result(const Variable& original, const Variable& result, std::string hook_name) {
  if (!original.options().type_equal(result.options())) {
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

void AutogradContext::set_materialize_grads(bool value) {
  materialize_grads_ = value;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_and_bump_dirty() const {
  for (auto& var : dirty_inputs_) {
    var->bump_version();
  }
  return dirty_inputs_;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_non_differentiable() const {
  return non_differentiable_;
}
}} // namespace torch::autograd
