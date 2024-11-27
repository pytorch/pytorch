#include <torch/csrc/autograd/functions/tensor.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace torch::autograd {

static variable_list CopyBackwards_apply_functional(
    variable_list&& grads,
    std::array<bool, 2> needs_input_grad,
    const c10::TensorOptions& src_options) {
  check_input_variables("CopyBackwards", grads, 1, -1, true);
  auto grad = c10::MaybeOwned<at::Tensor>::borrowed(grads[0]);
  variable_list grad_inputs(2);
  if (grad->defined()) {
    if (needs_input_grad[0]) {
      grad_inputs[0] = at::zeros_like(*grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (needs_input_grad[1]) {
      // Handle R->C copies without raising a warning
      const auto src_type = src_options.dtype().toScalarType();
      if (!c10::isComplexType(src_type) && grad->is_complex()) {
        grad = c10::MaybeOwned<at::Tensor>::owned(at::real(grads[0]));
      }

      at::DeviceGuard device_guard(src_options.device());
      grad_inputs[1] = grad->to(src_options);
    }
  }
  return grad_inputs;
}

ivalue_list CopyBackwards::retrieve_saved(SwapSavedVariables& saved) {
  saved.before(src_options);
  SavedState state;
  state.enqueue(src_options);
  saved.after(src_options);
  return state.stack;
}

c10::optional<functional_apply_t> CopyBackwards::get_functional() {
  auto needs_input_grad = std::array<bool, 2>{
      task_should_compute_output(0), task_should_compute_output(1)};
  return [needs_input_grad](
             const variable_list& inputs,
             const ivalue_list& stack) -> variable_list {
    SavedState state;
    state.stack = stack;
    at::TensorOptions src_options;
    state.dequeue(src_options);
    auto inputs_copy = inputs;

    return CopyBackwards_apply_functional(
        std::move(inputs_copy), needs_input_grad, src_options);
  };
}

auto CopyBackwards::apply(variable_list&& grads) -> variable_list {
  return CopyBackwards_apply_functional(
      std::move(grads),
      {task_should_compute_output(0), task_should_compute_output(1)},
      src_options);
}

void CopyBackwards::compiled_args(CompiledNodeArgs& args) {
  args.collect(src_options);
}
variable_list CopyBackwards::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  saved.before(src_options);
  auto result = apply(variable_list(inputs));
  saved.after(src_options);
  return result;
}

CopySlices::CopySlices(
    const Variable& base_var,
    at::TensorGeometry view_,
    std::unique_ptr<ViewFunc> view_fn_,
    std::shared_ptr<Node> fn_)
    : Node(),
      base(base_var),
      view(std::move(view_)),
      view_fn(std::move(view_fn_)),
      fn(std::move(fn_)) {
  // Take the next_edges of fn as our own, except for index 0 which goes
  // to base instead of the view.
  add_input_metadata(base_var);
  const auto num_outputs = fn->num_outputs();
  next_edges_.reserve(num_outputs);
  add_next_edge(impl::gradient_edge(base_var));
  for (const auto i : c10::irange(1, num_outputs)) {
    add_next_edge(fn->next_edge(i));
  }
}

template <typename F1>
static variable_list CopySlices_apply_functional(
    variable_list&& inputs,
    const std::vector<bool>& needs_input_grad,
    const at::TensorGeometry& base,
    const at::TensorGeometry& view,
    int64_t num_outputs,
    const F1& call_fn,
    const std::unique_ptr<ViewFunc>& view_fn) {
  auto& grad = inputs[0];

  auto result =
      grad.new_empty_strided_symint(base.sym_sizes(), base.sym_strides());
  result.copy_(grad);

  at::Tensor grad_slice;
  if (view_fn) {
    grad_slice = (*view_fn)(result);
  } else {
    auto offset = view.sym_storage_offset() - base.sym_storage_offset();
    grad_slice =
        result.as_strided_symint(view.sym_sizes(), view.sym_strides(), offset);
  }

  // TODO: We clone grad_slice because we modify it below and "fn" might save
  // it for the backward of res. We might be able to avoid the clone() if
  // double-backprop is disabled.
  auto res = call_fn({grad_slice.clone(at::MemoryFormat::Contiguous)});

  variable_list grad_inputs(num_outputs);
  for (const auto i : c10::irange(res.size())) {
    if (needs_input_grad[i]) {
      if (!res[i].defined()) {
        // If the output is not defined, treat it as if it was a zero tensor.
        // This can happen if users define a custom Function.
        continue;
      }
      if (i == 0) {
        grad_slice.copy_(res[i]);
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
        grad_inputs[i] = std::move(result); // NOLINT(bugprone-use-after-move)
      } else {
        grad_inputs[i] = std::move(res[i]);
      }
    }
  }
  return grad_inputs;
}

// common code between apply/apply_with_saved
template <typename T>
inline variable_list CopySlices::apply_impl(
    variable_list&& inputs,
    const T& call_fn) {
  check_input_variables("CopySlices", inputs, 1, -1, true);
  auto& grad = inputs[0];
  if (!grad.defined()) {
    return variable_list(num_outputs());
  }

  if (!fn) {
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

  // Acquire lock to here protect thread safety on fn
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

  // See Note [View + Inplace update for view tensor] For more details on this
  // block Since the gradient edge for the 0th input is different between `this`
  // and `fn`, make sure that the one from `fn` has the same metadata in the
  // current GraphTask's exec_info as the one on `this`.
  const auto exec_info = get_current_graph_task_exec_info();
  if (exec_info && !exec_info->empty()) {
    const auto& fn_edge = fn->next_edge(0);
    const auto& this_edge = this->next_edge(0);
    TORCH_INTERNAL_ASSERT(fn_edge.is_valid() == this_edge.is_valid());
    if (fn_edge.is_valid()) {
      const auto fn_next_node = fn_edge.function.get();
      auto it = exec_info->find(fn_next_node);
      if (it == exec_info->end()) {
        // Node is not in the exec_info already
        if (task_should_compute_output(0)) {
          // And we need gradient for the corresponding output
          add_node_to_current_graph_task_exec_info(fn_next_node);
          // There is no need to remove this after execution because we are
          // guaranteed that this->next_edge(0) must be in the history of
          // fn->next_edge(0) (we cannot easily assert this as it might be far
          // away if there were many chained views). This means that, since
          // fn->next_edge(0) was not needed (no exec_info entry for it), we
          // know that nothing downstream of fn->next_edge(0) is needed either
          // (otherwise the whole path from that Node to this->next_edge(0)
          // would be needed as well). This means that no other Node will ever
          // look at fn->next_edge(0) metadata and thus there is no need to
          // clean them up.
        }
      } else {
        TORCH_INTERNAL_ASSERT(
            it->second.should_execute() == task_should_compute_output(0));
      }
    }
  }

  // Sanity check that the graph was never modified after the fact (it is
  // read-only!)
  TORCH_INTERNAL_ASSERT(num_outputs() == fn->num_outputs());
  for (const auto i : c10::irange(1, this->num_outputs())) {
    TORCH_INTERNAL_ASSERT(
        fn->next_edge(i).function.get() == this->next_edge(i).function.get());
  }

  std::vector<bool> needs_input_grad;
  for (const auto i : c10::irange(num_outputs())) {
    needs_input_grad.emplace_back(task_should_compute_output(i));
  }

  return CopySlices_apply_functional(
      std::move(inputs),
      needs_input_grad,
      base,
      view,
      num_outputs(),
      call_fn,
      view_fn);
}

void CopySlices::release_variables() {
  // Acquire lock to here protect thread safety on fn
  std::lock_guard<std::mutex> lock(mutex_);
  fn = nullptr;
}

void CopySlices::compiled_args(CompiledNodeArgs& args) {
  TORCH_CHECK(!view_fn, "view_fn not supported by compiled autograd")
  TORCH_INTERNAL_ASSERT((bool)fn);
  args.collect(base);
  args.collect(view);
  args.collect(fn);
  fn->compiled_args(args);
}

variable_list CopySlices::apply_with_saved(
    const variable_list& grads,
    SwapSavedVariables& saved) {
  saved.before(base);
  saved.before(view);
  int call_count = 0;
  variable_list result = apply_impl(
      variable_list(grads),
      [this, &saved, &call_count](const variable_list& inputs2) {
        call_count++;
        return fn->apply_with_saved(inputs2, saved);
      });
  TORCH_INTERNAL_ASSERT(call_count == 1);
  saved.after(base);
  saved.after(view);
  return result;
}

ivalue_list CopySlices::retrieve_saved(SwapSavedVariables& saved) {
  saved.before(base);
  saved.before(view);

  SavedState state;
  state.enqueue(base);
  state.enqueue(view);

  auto fn_state = fn->retrieve_saved(saved);
  state.stack.insert(state.stack.end(), fn_state.begin(), fn_state.end());

  saved.after(base);
  saved.after(view);

  return state.stack;
}

c10::optional<functional_apply_t> CopySlices::get_functional() {
  TORCH_INTERNAL_ASSERT(
      !view_fn, "NYI: compiled autograd with CopySlices with view_fn");
  auto num_out = num_outputs();
  std::vector<bool> needs_input_grad;
  for (const auto i : c10::irange(num_outputs())) {
    needs_input_grad.emplace_back(task_should_compute_output(i));
  }
  auto fn2 = fn;

  return [fn2, num_out, needs_input_grad](
             const variable_list& inputs,
             const std::vector<c10::IValue>& saved) -> variable_list {
    SavedState state;
    state.stack = saved;
    at::TensorGeometry base;
    at::TensorGeometry view;
    state.dequeue(base);
    state.dequeue(view);

    // TODO(rzou): somehow we need to restore the state...
    auto call_fn = [fn2](variable_list&& inputs2) -> variable_list {
      return (*fn2)(std::move(inputs2));
    };
    // TODO(rzou): wut
    variable_list copied_inputs = inputs;

    return CopySlices_apply_functional(
        std::move(copied_inputs),
        needs_input_grad,
        base,
        view,
        num_out,
        call_fn,
        {});
  };
}

auto CopySlices::apply(variable_list&& inputs1) -> variable_list {
  return apply_impl(std::move(inputs1), [this](variable_list&& inputs2) {
    return (*fn)(std::move(inputs2));
  });
}

} // namespace torch::autograd
