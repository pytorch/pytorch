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

using torch::dynamo::autograd::IValuePacker;

static variable_list CopyBackwards_apply_functional(
    variable_list&& grads,
    std::array<bool, 2> needs_input_grad,
    const c10::TensorOptions& src_options) {
  check_input_variables("CopyBackwards", grads, 1, -1, true);
  auto& grad = std::move(grads)[0];
  variable_list grad_inputs(2);
  if (grad.defined()) {
    if (needs_input_grad[0]) {
      grad_inputs[0] = at::zeros_like(grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (needs_input_grad[1]) {
      // Handle R->C copies without raising a warning
      const auto src_type = src_options.dtype().toScalarType();
      if (!c10::isComplexType(src_type) && grad.is_complex()) {
        grad = at::real(grad);
      }

      at::DeviceGuard device_guard(src_options.device());
      grad_inputs[1] = grad.to(src_options);
    }
  }
  return grad_inputs;
}

static variable_list CopyBackwards_apply_functional_ivalue(
    const variable_list& grads,
    const ivalue_list& args) {
  PackedArgs r(args);
  auto needs_input_grad = r.unpack<std::array<bool, 2>>();
  auto src_options = r.unpack<c10::TensorOptions>();
  return CopyBackwards_apply_functional(
      variable_list(grads), needs_input_grad, src_options);
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

  static c10::once_flag flag;
  c10::call_once(flag, [&]() {
    std::vector<at::TypePtr> schema = {
        IValuePacker<std::array<bool, 2>>::packed_type(),
        IValuePacker<c10::TensorOptions>::packed_type()};
    const auto& interface = torch::dynamo::autograd::getPyCompilerInterface();
    interface->bind_function(
        saved.get_py_compiler(),
        name(),
        CopyBackwards_apply_functional_ivalue,
        schema);
  });

  PackedArgs packed_args;
  packed_args.pack<std::array<bool, 2>>(
      {task_should_compute_output(0), task_should_compute_output(1)});
  packed_args.pack(src_options);

  auto output_metadata = torch::dynamo::autograd::
      IValuePacker<std::vector<std::optional<InputMetadata>>>::pack(
          torch::dynamo::autograd::get_input_metadata(next_edges()));

  const auto& interface = torch::dynamo::autograd::getPyCompilerInterface();
  auto result = interface->call_function(
      saved.get_py_compiler(),
      "apply_functional",
      name(),
      inputs,
      std::move(packed_args).vec(),
      output_metadata);

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

void CopySlices::update_exec_info() {
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
}

// common code between apply/apply_with_saved
template <typename T>
inline variable_list CopySlices::apply_impl(
    variable_list&& inputs,
    const T& call_fn) {
  check_input_variables("CopySlices", inputs, 1, -1, true);
  auto& grad = std::move(inputs)[0];
  if (!grad.defined()) {
    return variable_list(num_outputs());
  }

  // Acquire lock to here protect thread safety on fn
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

  if (!fn) {
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

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

  update_exec_info();

  // TODO: We clone grad_slice because we modify it below and "fn" might save
  // it for the backward of res. We might be able to avoid the clone() if
  // double-backprop is disabled.
  auto res = call_fn({grad_slice.clone(at::MemoryFormat::Contiguous)});

  variable_list grad_inputs(num_outputs());
  for (const auto i : c10::irange(res.size())) {
    if (task_should_compute_output(i)) {
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

  auto results = variable_list(num_outputs());
  if (grads[0].defined()) {
    if (!fn) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    update_exec_info();

    std::vector<bool> needs_input_grad;
    for (const auto i : c10::irange(num_outputs())) {
      needs_input_grad.emplace_back(task_should_compute_output(i));
    }
    // Not yet supported, also doesn't happen in typical eager mode execution
    // (this only happens by default with torch-xla).
    TORCH_INTERNAL_ASSERT(!view_fn);
    const auto& interface = torch::dynamo::autograd::getPyCompilerInterface();
    variable_list stuff = interface->call_copy_slices_prologue(
        saved.get_py_compiler(), grads, base, view);
    TORCH_INTERNAL_ASSERT(stuff.size() == 3);
    // These variables are named the same as in CopySlices::apply_impl.
    // Follow along there.
    auto result = stuff[0];
    auto grad_slice = stuff[1];
    auto grad_slice_clone = stuff[2];
    auto res = fn->apply_with_saved({grad_slice_clone}, saved);
    results = interface->call_copy_slices_epilogue(
        saved.get_py_compiler(), needs_input_grad, result, res, grad_slice);
  }

  saved.after(base);
  saved.after(view);
  return results;
}

auto CopySlices::apply(variable_list&& inputs1) -> variable_list {
  return apply_impl(std::move(inputs1), [this](variable_list&& inputs2) {
    return (*fn)(std::move(inputs2));
  });
}

} // namespace torch::autograd
