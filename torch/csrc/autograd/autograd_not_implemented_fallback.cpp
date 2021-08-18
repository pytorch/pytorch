#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

#include <c10/util/irange.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>

namespace torch { namespace autograd {

namespace {

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

}

void autogradNotImplementedFallbackImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // Mimics the logic of a VariableType NotImplemented kernel
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto& arguments = schema.arguments();
  const auto& returns = schema.returns();
  const auto num_arguments = arguments.size();
  const auto num_returns = returns.size();
  const auto stack_start = stack->size() - num_arguments;
  const bool grad_mode = GradMode::is_enabled();
  std::vector<const at::Tensor*> tensors_requiring_grad_on_stack;

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

  #ifndef NDEBUG
  int aliased_input_idx = -1;
  int aliased_output_idx = -1;
  for (const auto i : c10::irange(num_returns)) {
    const auto& alias_info = returns[i].alias_info();
    if (alias_info.has_value()) {
      AT_ASSERT(aliased_output_idx == -1); // Assume only a single aliased output
      aliased_output_idx = i;
    }
  }
  for (const auto i : c10::irange(num_arguments)) {
    const auto& alias_info = arguments[i].alias_info();
    if (alias_info.has_value()) {
      AT_ASSERT(aliased_input_idx == -1); // Assume only a single aliased input
      aliased_input_idx = i;
    }
  }
  #endif

  size_t num_tensor_inputs = 0;  // Only used for DEBUG-only checks

  _foreach_tensor([&](size_t _, size_t idx_arg, const at::Tensor& t) {
    if (arguments[idx_arg].type()->kind() != c10::TypeKind::OptionalType) {
      TORCH_CHECK(t.defined(), "Expected argument ", idx_arg, " of ", op_name, " to be defined.");
    }
    if (grad_mode && t.requires_grad()) {
      tensors_requiring_grad_on_stack.push_back(&t);
    }
    num_tensor_inputs++;
    TORCH_CHECK_NOT_IMPLEMENTED(!isFwGradDefined(t), "Trying to use forward AD with ", op_name, " that does not support it.");
  }, stack, stack_start, num_arguments);

  const bool any_requires_grad = tensors_requiring_grad_on_stack.size() > 0;

  _foreach_tensor([&](size_t _, size_t i, const at::Tensor& t) {
    const auto& alias_info = arguments[i].alias_info();
    if (alias_info.has_value() && alias_info->isWrite()) {
      check_inplace(t, any_requires_grad);
    }
  }, stack, stack_start, num_arguments);

  std::shared_ptr<NotImplemented> grad_fn;
  if (any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented(op_name), deleteNode);
    grad_fn->set_next_edges(collect_next_edges(tensors_requiring_grad_on_stack));
  }

  #ifndef NDEBUG
  // See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
  std::vector<c10::IValue> stack_copy(*stack);
  std::vector<c10::intrusive_ptr<c10::TensorImpl>> impl_saved;
  impl_saved.reserve(num_tensor_inputs);
  std::vector<c10::optional<c10::Storage>> storage_saved;
  storage_saved.reserve(num_tensor_inputs);
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    storage_saved.push_back(t.has_storage() ? c10::optional<c10::Storage>(t.storage()) : c10::nullopt);
    impl_saved.push_back(t.getIntrusivePtr());
  }, &stack_copy, stack_start, num_arguments);
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
  }
  #ifndef NDEBUG
  _foreach_tensor([&](size_t idx_tensor, size_t _, const at::Tensor& t) {
    if (storage_saved.at(idx_tensor).has_value())
      AT_ASSERT(storage_saved.at(idx_tensor).value().is_alias_of(t.storage()));
    if (impl_saved.at(idx_tensor))
      AT_ASSERT(impl_saved.at(idx_tensor) == t.getIntrusivePtr());
  }, &stack_copy, stack_start, num_arguments);
  _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
    if (!is_inplace_output[idx_ret])
      AT_ASSERT(t.use_count() <= 1);  // Okay to return undefined tensor
    if (!is_aliased_output[idx_ret] && t.has_storage())
      AT_ASSERT(t.storage().use_count() == 1);
  }, stack, stack->size() - num_returns, num_returns);
  if (aliased_input_idx != -1 && aliased_output_idx != -1) {
    const c10::IValue& aliased_input_iv = stack_copy[stack_start + aliased_input_idx];
    const c10::IValue& aliased_output_iv = (*stack)[stack->size() - num_returns + aliased_output_idx];
    // We do not support views embedded inside tensorlist
    AT_ASSERT(aliased_input_iv.isTensor());
    AT_ASSERT(aliased_output_iv.isTensor());
    const at::Tensor& aliased_input = aliased_input_iv.toTensor();
    const at::Tensor& aliased_output = aliased_input_iv.toTensor();
    if(is_aliased_output[aliased_input_idx] && aliased_input.has_storage())
      AT_ASSERT(aliased_input.storage().is_alias_of(aliased_output.storage()));
  }
  #endif

  if (any_requires_grad) {
    _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
      if (isDifferentiableType(t.scalar_type())) {
        if (is_inplace_output[idx_ret]) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          rebase_history(const_cast<at::Tensor&>(t), grad_fn);
        } else {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          set_history(const_cast<at::Tensor&>(t), grad_fn);
        }
      }
    }, stack, stack->size() - num_returns, num_returns);
  }
}

torch::CppFunction autogradNotImplementedFallback() {
  return torch::CppFunction::makeFromBoxedFunction<&autogradNotImplementedFallbackImpl>();
}

}} // namespace torch::autograd
