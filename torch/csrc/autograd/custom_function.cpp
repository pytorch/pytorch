#include <c10/util/irange.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

#include <utility>

namespace torch {
namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
    : layout(var.layout()),
      device(var.device()),
      scalar_type(var.scalar_type()),
      size(var.sym_sizes().vec()),
      requires_grad(var.requires_grad()),
      is_empty(false) {}

VariableInfo::VariableInfo() : requires_grad(false), is_empty(true) {}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  if (is_empty) {
    // Return undefined tensor.
    return at::Tensor();
  } else {
    return at::zeros_symint(
        size, at::TensorOptions(scalar_type).device(device).layout(layout));
  }
}

// This function has two main goals:
//  1) Use the user-provided jvp function to populate the outputs' forward
//  gradient 2) Perform error checking to ensure that view and inplace ops are
//  properly handled
//
// For 1) we have to:
//  - Create a variable_list of grad_inputs based on the function inputs
//  - Call the user jvp function with these to get the grad_outputs
//  - Set the forward grad field on each output based on these grad_outputs
//
// For 2) we want to check the following:
//  - If an output is a view, then the generated forward grad must be a view as
//  well and
//    the output's base's forward grad must be the output's forward grad's base.
//  - If an input was modified inplace (it must be an output as well) we make
//  sure that its
//    forward grad was also modified inplace and already present on the
//    corresponding output.
static void _process_forward_mode_AD(
    const variable_list& inputs,
    std::unordered_map<at::TensorImpl*, size_t> inputs_mapping,
    const at::ArrayRef<c10::optional<Variable>> raw_outputs,
    const optional_variable_list& outputs,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    _jvp_fn_t jvp_user_function) {
  // TODO handle multiple levels here
  uint64_t level = 0;

  const auto num_inputs = inputs.size();
  const auto num_outputs = outputs.size();

  // The tracking info below are used to perform the view and inplace checks.
  // They are lazily initialized to reduce the cost of this function in the
  // common case where the user is not using forward mode AD.
  variable_list input_grads;
  std::vector<int64_t> grad_versions;
  std::vector<at::TensorImpl*> grad_impls;
  std::unordered_map<at::TensorImpl*, size_t> inputs_bases;

  auto init_tracked_info = [&]() {
    input_grads.resize(num_inputs);
    grad_versions.resize(num_inputs);
    grad_impls.resize(num_inputs);

    for (const auto i : c10::irange(num_inputs)) {
      const auto& inp = inputs[i];
      if (inp.is_view() && impl::get_view_autograd_meta(inp)->has_fw_view()) {
        inputs_bases.emplace(
            impl::get_view_autograd_meta(inp)
                ->get_forward_view()
                .base_.unsafeGetTensorImpl(),
            i);
      } else {
        inputs_bases.emplace(inp.unsafeGetTensorImpl(), i);
      }
    }
  };

  bool any_input_has_grad = false;
  // Extract the input's forward gradients and record any info we will need
  // later
  for (const auto i : c10::irange(num_inputs)) {
    const auto& inp = inputs[i];
    if (!inp.defined()) {
      continue;
    }
    const auto& fw_grad = inp._fw_grad(level);
    if (fw_grad.defined()) {
      if (!any_input_has_grad) {
        any_input_has_grad = true;
        init_tracked_info();
      }
      input_grads[i] = fw_grad;
      grad_versions[i] = fw_grad._version();
      grad_impls[i] = fw_grad.unsafeGetTensorImpl();
    }
  }

  // If no input has forward grad, nothing to do here
  if (!any_input_has_grad) {
    return;
  }

  torch::autograd::variable_list forward_grads;
  {
    at::AutoFwGradMode fw_grad_mode(false);
    forward_grads = jvp_user_function(inputs, std::move(input_grads));
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const auto num_forward_grads = forward_grads.size();
  // contrary to backward mode, we don't allow returning too many gradients
  TORCH_CHECK(
      num_forward_grads == num_outputs,
      "Function's jvp returned "
      "an invalid number of forward gradients (expected ",
      num_outputs,
      " but got ",
      num_forward_grads,
      ")");

  for (const auto i : c10::irange(num_outputs)) {
    const auto& out =
        outputs[i].has_value() ? outputs[i].value() : at::Tensor();
    auto out_tensor_impl = raw_outputs[i].value().unsafeGetTensorImpl();
    bool is_differentiable =
        (non_differentiable.count(out_tensor_impl) == 0 &&
         isDifferentiableType(raw_outputs[i].value().scalar_type()));
    const auto& out_grad = forward_grads[i];
    if (!out.defined() || !is_differentiable) {
      TORCH_CHECK(
          !out_grad.defined(),
          "Function's jvp returned a gradient at position ",
          i,
          ", but "
          " the corresponding forward output is not a differentiable Tensor."
          "You should return None at that position instead.");
      continue;
    }

    TORCH_INTERNAL_ASSERT(raw_outputs[i].has_value());
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;

    if (is_modified) {
      TORCH_CHECK(
          is_input,
          "Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
          " is no need to pass it to mark_dirty().");
      auto inp_idx = inputs_mapping[out_tensor_impl];
      if (grad_impls[inp_idx]) {
        // If there was already a forward grad for that input
        // Just make sure that it is modified inplace and returned as-is
        TORCH_CHECK(
            out_grad._version() != grad_versions[inp_idx],
            "An inplace custom Function is not modifying the "
            "forward mode gradients inplace. If the forward is modifying an input inplace, then the jvp "
            "function must modify the corresponding gradient inplace.")
        TORCH_CHECK(
            out_grad.unsafeGetTensorImpl() == grad_impls[inp_idx],
            "An inplace custom Function is not returning the "
            "forward mode gradients as-is. If the forward is modifying an input inplace, then the jvp "
            "function must modify the gradient inplace and return it as-is.")
      } else {
        // If that Tensor didn't had gradients already, set the newly returned
        // one We could also use inputs[inp_idx] here as it is the same as out
        out._set_fw_grad(out_grad, level, /* is_inplace_op */ true);
      }
    } else {
      // At this point, outputs[i] cannot be one of the input (raw_outputs[i]
      // might be but was changed by the backward code)
      TORCH_INTERNAL_ASSERT(
          inputs_mapping.count(out.unsafeGetTensorImpl()) == 0);

      if (out.is_view() && impl::get_view_autograd_meta(out)->has_fw_view()) {
        // If the output is a view
        const auto& out_view_info =
            impl::get_view_autograd_meta(out)->get_forward_view();
        if (inputs_bases.count(out_view_info.base_.unsafeGetTensorImpl())) {
          // And it is a view of an input (either that input is its base or they
          // have a common base)
          const auto matching_input_idx =
              inputs_bases[out_view_info.base_.unsafeGetTensorImpl()];
          const auto& matching_input = inputs[matching_input_idx];

          const auto& matching_input_grad = matching_input._fw_grad(level);

          // If the matching input has a forward grad, the user should have
          // returned a view of that Tensor
          if (matching_input_grad.defined()) {
            TORCH_CHECK(
                out_grad.is_view() &&
                    impl::get_view_autograd_meta(out_grad)->has_fw_view(),
                "A custom Function's forward is returning a view (or an input as-is) but the jvp is not "
                "returning a view.");
            const auto& out_grad_base = impl::get_view_autograd_meta(out_grad)
                                            ->get_forward_view()
                                            .base_;
            if (matching_input_grad.is_view() &&
                impl::get_view_autograd_meta(matching_input_grad)
                    ->has_fw_view()) {
              // If the matching input's grad is a view, ensure that the
              // out_grad is a view of the same base
              const auto& matching_input_grad_base =
                  impl::get_view_autograd_meta(matching_input_grad)
                      ->get_forward_view()
                      .base_;
              TORCH_CHECK(
                  matching_input_grad_base.unsafeGetTensorImpl() ==
                      out_grad_base.unsafeGetTensorImpl(),
                  "A custom Function is returning a view but the jvp is not returning a view of the same base as "
                  "the given grad input.");
            } else {
              // If the matching input's grad is not a view, then it must be the
              // output gradient's base
              TORCH_CHECK(
                  matching_input_grad.unsafeGetTensorImpl() ==
                      out_grad_base.unsafeGetTensorImpl(),
                  "A custom Function is returning a view but the jvp is not returning a view of the given grad input.");
            }
          } else {
            // We have a view op where the input didn't have a forward grad but
            // the user returned one for the output To ensure that we maintain
            // the view/inplace constraints, we consider this as an inplace op
            // This case CANNOT happen in codegen as all view ops are mapping
            // from one Tensor to one Tensor and so the output of the view
            // cannot have a forward grad if the base does not.
            out._set_fw_grad(out_grad, level, /* is_inplace_op */ true);
            return;
          }
        }
      }

      out._set_fw_grad(out_grad, level, /* is_inplace_op */ false);
    }
  }
}

static at::Tensor _view_as_self_with_no_grad(at::Tensor self) {
  // This is called below in _process_backward_mode_ad in two places:
  //
  // (1) An input has been returned, but it wasn't modified. Return it as a view
  // so that we can attach a new grad_fn to the Variable.
  // Run in no_grad mode to mimic the behavior of the forward.
  //
  // (2) Though it is not necessary for the purposes of attaching grad_fn, we
  // also call this function when an output is non-differentiable (and does not
  // require grad). to help custom forward AD UX more consistent. We'd like to
  // uniformly say that returning an input as-is is treated as if
  // `self.view_as(self)` were returned for that output.
  //
  // Alternatively, we could have not disabled forward grad while performing
  // this view, but it would mean that the user defined jvp may be silently
  // ignored.
  at::AutoFwGradMode fw_grad_mode(false);
  AutoGradMode grad_mode(false);
  return self.view_as(self);
}

static optional_variable_list _process_backward_mode_ad(
    const std::unordered_map<at::TensorImpl*, size_t>& inputs_mapping,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    const at::ArrayRef<c10::optional<Variable>> raw_outputs,
    const std::shared_ptr<Node>& cdata,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context) {
  int num_outputs = raw_outputs.size();

  const char* error_msg_input_returned_as_is =
      "A input that has been returned as-is as output is being saved for backward. "
      "This is not supported if you override setup_context. You should return and "
      "save a view of the input instead, e.g. with x.view_as(x) or setup ctx inside "
      "the forward function itself.";

  // Sets the grad_fn and output_nr of an output Variable.
  auto set_history = [&](Variable& var,
                         uint32_t output_nr,
                         bool is_input,
                         bool is_modified,
                         bool is_differentiable,
                         bool is_saved_and_setup_context) {
    if (!is_differentiable) {
      if (!var.requires_grad()) {
        if (is_input && !is_modified) {
          TORCH_CHECK(
              !is_saved_and_setup_context, error_msg_input_returned_as_is)
          var = _view_as_self_with_no_grad(var);
        }
        return;
      }
      // Return detached aliases of inputs, instead of changing their
      // requires_grad property.
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
        TORCH_CHECK(
            false,
            "a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // No need to mark as modified Tensors that are not inputs.
      if (!is_input) {
        TORCH_WARN(
            "Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
            " is no need to pass it to mark_dirty().");
      }
      // If the input is a view, the rebase will need to rewrite the graph and
      // this only works if we have a single output to this Function.
      TORCH_CHECK(
          !(var.is_view() && num_outputs > 1),
          "If your Function modifies inplace an input that is a view"
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
        auto& grad_acc = dynamic_cast<AccumulateGrad&>(*grad_acc_fn);
        grad_acc.variable.reset();
      }
      if (cdata) {
        impl::rebase_history(var, {cdata, output_nr});
      }
    } else if (is_input) {
      TORCH_CHECK(!is_saved_and_setup_context, error_msg_input_returned_as_is)
      var = _view_as_self_with_no_grad(var);
      impl::set_gradient_edge(var, {cdata, output_nr});
    } else if (cdata) {
      impl::set_gradient_edge(var, {cdata, output_nr});
    }
  };

  optional_variable_list outputs;
  std::unordered_set<at::TensorImpl*> outputs_impl; // For dirty_inputs check
  outputs.reserve(num_outputs);
  int num_diff_outputs = 0;

  for (const auto i : c10::irange(num_outputs)) {
    // We put a undefined_input placeholder for outputs that are not tensor and
    // for when the output tensor is not differentiable (see below)
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
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    bool is_differentiable = cdata &&
        non_differentiable.count(out_tensor_impl) == 0 &&
        isDifferentiableType(var.scalar_type());
    bool is_saved_and_setup_context =
        to_save_if_setup_context.count(out_tensor_impl) > 0;

    if (cdata) {
      auto output_nr = -1;
      if (!is_differentiable) {
        output_nr = cdata->add_input_metadata(Node::undefined_input());
      } else {
        output_nr = cdata->add_input_metadata(var);
      }
      AT_ASSERT(i == (int)output_nr);
    }
    set_history(
        var,
        i,
        is_input,
        is_modified,
        is_differentiable,
        is_saved_and_setup_context);

    // For deprecation cycle. Can be removed after 1.6. In the case where we
    // detected a view in no grad mode during the forward, only warn the user
    // (do not change the flag if we return and input that is a view as is). See
    // NOTE [ View + Inplace detection ] for why we replace everything by a
    // warning.
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

  // If multiple differentiable outputs are returned, we do not allow views to
  // be modified inplace See NOTE [ View + Inplace detection ] for more details
  if (num_diff_outputs > 1) {
    for (auto& var : outputs) {
      if (var.has_value()) {
        auto diff_view_meta = impl::get_view_autograd_meta(var.value());
        if (diff_view_meta && diff_view_meta->has_bw_view()) {
          diff_view_meta->set_creation_meta(CreationMeta::MULTI_OUTPUT_NODE);
        }
      }
    }
  }

  // All the modified Tensors must be returned as is for the rewrite to be
  // valid.
  for (auto& dirty_input : dirty_inputs) {
    TORCH_CHECK(
        outputs_impl.count(dirty_input) > 0,
        "Some elements marked as dirty during the forward method were not returned as output. The"
        " inputs that are modified inplace must all be outputs of the Function.");
  }

  return outputs;
}

optional_variable_list _wrap_outputs(
    const variable_list& input_vars,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    const at::ArrayRef<c10::optional<Variable>> raw_outputs,
    const std::shared_ptr<Node>& cdata,
    _jvp_fn_t jvp_user_function,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context) {
  std::unordered_map<at::TensorImpl*, size_t> inputs_mapping;
  inputs_mapping.reserve(input_vars.size());
  for (const auto i : c10::irange(input_vars.size())) {
    inputs_mapping.emplace(input_vars[i].unsafeGetTensorImpl(), i);
  }

  auto outputs = _process_backward_mode_ad(
      inputs_mapping,
      non_differentiable,
      dirty_inputs,
      raw_outputs,
      cdata,
      to_save_if_setup_context);

  // This must happen after the backward processing as we expect the
  // computations happening here to track backward mode gradients.
  _process_forward_mode_AD(
      input_vars,
      std::move(inputs_mapping),
      raw_outputs,
      outputs,
      non_differentiable,
      dirty_inputs,
      std::move(jvp_user_function));

  return outputs;
}

void check_variable_result(
    const at::TensorBase& original,
    const at::TensorBase& result,
    std::string hook_name) {
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

  if (original.sym_sizes().vec() != result.sym_sizes().vec()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the size of value";
    throw std::runtime_error(ss.str());
  }
}

void AutogradContext::save_for_backward(variable_list to_save) {
  to_save_ = std::move(to_save);
}

// The logic for handling saved variables here is the same as
// python_function.cpp See _save_variables() and unpack_saved_variables()
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

bool AutogradContext::needs_input_grad(size_t output_edge_index) const {
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  return ptr->task_should_compute_output(output_edge_index);
}

bool AutogradContext::needs_input_grad(
    std::initializer_list<IndexRange> idxs) const {
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  return ptr->task_should_compute_output(idxs);
}

void AutogradContext::mark_dirty(const variable_list& inputs) {
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  for (auto& var : inputs) {
    dirty_inputs_.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::mark_non_differentiable(const variable_list& outputs) {
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  for (auto& var : outputs) {
    non_differentiable_.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::set_materialize_grads(bool value) {
  materialize_grads_ = value;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_and_bump_dirty()
    const {
  for (auto& var : dirty_inputs_) {
    var->bump_version();
  }
  return dirty_inputs_;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::
    get_non_differentiable() const {
  return non_differentiable_;
}
} // namespace autograd
} // namespace torch
