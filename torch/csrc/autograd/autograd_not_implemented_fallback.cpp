#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

#include <c10/util/irange.h>

#include <ATen/core/TorchDispatchUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <vector>

namespace torch {
namespace autograd {

namespace {

template <typename F>
void _foreach_tensor(
    F fn,
    torch::jit::Stack* stack,
    size_t stack_start,
    size_t size) {
  // Enumerate over tensors in a stack, including ones in TensorLists
  int idx_tensor = 0;
  for (const auto idx_arg : c10::irange(size)) {
    auto& ivalue = (*stack)[stack_start + idx_arg];
    if (ivalue.isTensor()) { // true for optional tensor that has value
      const auto& tensor = ivalue.toTensor();
      fn(idx_tensor, idx_arg, tensor);
      idx_tensor++;
    } else if (ivalue.isTensorList()) {
      for (const auto& iv : ivalue.toListRef()) {
        const auto& tensor = iv.toTensor();
        fn(idx_tensor, idx_arg, tensor);
        idx_tensor++;
      }
    }
  }
}

} // namespace

void autogradNotImplementedFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // Mimics a subset of the logic of a VariableType NotImplemented kernel
  // See gen_variable_type.py
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();
  const auto stack_start = stack->size() - num_arguments;
  const bool grad_mode = GradMode::is_enabled();
  std::vector<const at::Tensor*> tensors_requiring_grad_on_stack;

  // Keep track of which outputs are output of in-place modification
  // so we can rebase_history if necessary
  std::vector<bool> is_inplace_output(num_returns, false);
  bool any_is_inplace_output = false;
  std::vector<bool> is_aliased_output(num_returns, false);
  int aliased_output_idx = -1;

  for (const auto i : c10::irange(num_returns)) {
    if (schema.is_aliasing({c10::SchemaArgType::output, i})) {
      if (schema.is_mutable({c10::SchemaArgType::output, i})) {
        is_inplace_output[i] = true;
        any_is_inplace_output = true;
      } else {
        TORCH_CHECK(
            aliased_output_idx == -1,
            "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
            "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
            "Please rewrite your function as a composite function.");
        aliased_output_idx = i;
      }
      is_aliased_output[i] = true;
    }
  }

  int aliased_input_idx = -1;
  for (const auto i : c10::irange(num_arguments)) {
    if (schema.is_aliasing({c10::SchemaArgType::input, i}) &&
        !schema.is_mutable({c10::SchemaArgType::input, i})) {
      TORCH_CHECK(
          aliased_input_idx == -1,
          "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
          "Please rewrite your function as a composite function.");
      aliased_input_idx = i;
    }
  }

  size_t num_tensor_inputs = 0; // Only used for DEBUG-only checks
  _foreach_tensor(
      [&](size_t _, size_t idx_arg, const at::Tensor& t) {
        if (grad_mode && t.requires_grad()) {
          tensors_requiring_grad_on_stack.push_back(&t);
        }
        num_tensor_inputs++;
        TORCH_CHECK_NOT_IMPLEMENTED(
            !isFwGradDefined(t),
            "Trying to use forward AD with ",
            op_name,
            " that does not support it.");
      },
      stack,
      stack_start,
      num_arguments);

  const bool any_requires_grad = tensors_requiring_grad_on_stack.size() > 0;

  _foreach_tensor(
      [&](size_t _, size_t i, const at::Tensor& t) {
        if (schema.is_mutable({c10::SchemaArgType::input, i})) {
          check_inplace(t, any_requires_grad);
        }
      },
      stack,
      stack_start,
      num_arguments);

  std::shared_ptr<NotImplemented> grad_fn;
  if (any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(
        new NotImplemented(op_name), deleteNode);
    grad_fn->set_next_edges(
        collect_next_edges(tensors_requiring_grad_on_stack));
  }

#ifndef NDEBUG
  // See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
  auto stack_args_copy =
      std::vector<c10::IValue>(stack->begin() + stack_start, stack->end());
  std::vector<c10::intrusive_ptr<c10::TensorImpl>> impl_saved;
  impl_saved.reserve(num_tensor_inputs);
  std::vector<c10::optional<c10::Storage>> storage_saved;
  storage_saved.reserve(num_tensor_inputs);
  _foreach_tensor(
      [&](size_t idx, size_t _, const at::Tensor& t) {
        storage_saved.push_back(
            t.has_storage() ? c10::optional<c10::Storage>(t.storage())
                            : c10::nullopt);
        impl_saved.push_back(t.getIntrusivePtr());
      },
      &stack_args_copy,
      0,
      num_arguments);
#endif
  if (aliased_input_idx != -1 || any_is_inplace_output) {
    at::AutoDispatchBelowAutograd guard;
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
  } else {
    // If neither in-place nor view
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatchBoxed(
        dispatch_keys & c10::after_ADInplaceOrView_keyset, stack);
  }
#ifndef NDEBUG
  _foreach_tensor(
      [&](size_t idx_tensor, size_t _, const at::Tensor& t) {
        if (storage_saved.at(idx_tensor).has_value())
          TORCH_INTERNAL_ASSERT(
              storage_saved.at(idx_tensor).value().is_alias_of(t.storage()),
              op_name);
        if (impl_saved.at(idx_tensor))
          TORCH_INTERNAL_ASSERT(
              impl_saved.at(idx_tensor) == t.getIntrusivePtr(), op_name);
      },
      &stack_args_copy,
      0,
      num_arguments);
  _foreach_tensor(
      [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
        if (at::impl::tensor_has_dispatch(t) ||
            at::impl::dispatch_mode_enabled())
          return;
        if (!is_inplace_output[idx_ret])
          TORCH_INTERNAL_ASSERT(
              t.use_count() <= 1, op_name); // Okay to return undefined tensor
        if (!is_aliased_output[idx_ret] && t.has_storage())
          TORCH_INTERNAL_ASSERT(t.storage().use_count() == 1);
      },
      stack,
      stack->size() - num_returns,
      num_returns);
  // There should be only a single base-view pair, make sure their storage is
  // aliased.
  if (aliased_input_idx != -1 && aliased_output_idx != -1) {
    const c10::IValue& aliased_input_iv = stack_args_copy[aliased_input_idx];
    const c10::IValue& aliased_output_iv =
        (*stack)[stack->size() - num_returns + aliased_output_idx];
    TORCH_INTERNAL_ASSERT(aliased_input_iv.isTensor(), op_name);
    TORCH_INTERNAL_ASSERT(
        aliased_output_iv.isTensor() || aliased_output_iv.isTensorList(),
        op_name);
    const at::Tensor& aliased_input = aliased_input_iv.toTensor();
    if (aliased_input.has_storage()) {
      if (aliased_output_iv.isTensor()) {
        const at::Tensor& aliased_output = aliased_input_iv.toTensor();
        TORCH_INTERNAL_ASSERT(
            aliased_input.storage().is_alias_of(aliased_output.storage()),
            op_name);
      } else {
        const auto aliased_output_vec = aliased_output_iv.toTensorVector();
        for (const auto& aliased_output : aliased_output_vec) {
          TORCH_INTERNAL_ASSERT(
              aliased_input.storage().is_alias_of(aliased_output.storage()),
              op_name);
        }
      }
    }
  }
#endif

  if (any_requires_grad) {
    _foreach_tensor(
        [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
          if (isDifferentiableType(t.scalar_type())) {
            if (is_inplace_output[idx_ret]) {
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              rebase_history(const_cast<at::Tensor&>(t), grad_fn);
            } else {
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              set_history(const_cast<at::Tensor&>(t), grad_fn);
            }
          }
        },
        stack,
        stack->size() - num_returns,
        num_returns);
  }
}

torch::CppFunction autogradNotImplementedFallback() {
  return torch::CppFunction::makeFromBoxedFunction<
      &autogradNotImplementedFallbackImpl>();
}

void autogradNotImplementedInplaceOrViewFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // Mimics a subset of the logic from ADInplaceOrViewType kernel:
  // - see gen_inplace_or_view_type.py
  // - this should only be used with autogradNotImplementedFallback above
  // - For more information see
  // https://pytorch.org/tutorials/advanced/dispatcher
  //
  // NOTE [ Limitations of ADInplaceOrView boxed kernel ]
  //
  // This op should only be used with autogradNotImplementedFallback kernel
  // because there is some logic we need specifically to enforce that even
  // if we do in-place on view's created in this kernel, the proper "derivative
  // is not implemented" error is still raised.
  //
  // Just like the codegened kernel, we try to enforce some things:
  // - For views: we enforce that the view relationship is between the first
  // input
  //   and the first output (which may be either Tensor or vec of Tensors
  // - For inplace (TODO?): enforce that the same op cannot be both a view and
  // inplace
  //   that is not allowed in the gen_inplace_or_view logic
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();
  const auto stack_start = stack->size() - num_arguments;

  at::Tensor aliased_input;

  int64_t aliased_output_idx = -1;
  for (const auto i : c10::irange(num_returns)) {
    if (schema.is_aliasing({c10::SchemaArgType::output, i}) &&
        !schema.is_mutable({c10::SchemaArgType::output, i})) {
      TORCH_CHECK(
          aliased_output_idx == -1,
          "Fallback ADInplaceOrView kernel expects only a single output in the operator schema to have a "
          "non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
          "Please rewrite your function as a composite function.");
      aliased_output_idx = i;
    }
  }

  int64_t aliased_input_idx = -1;
  for (const auto i : c10::irange(num_arguments)) {
    if (schema.is_aliasing({c10::SchemaArgType::input, i}) &&
        !schema.is_mutable({c10::SchemaArgType::input, i})) {
      TORCH_CHECK(
          aliased_input_idx == -1,
          "Fallback ADInplaceOrView kernel expects only a single input in the operator schema to have a "
          "non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
          "Please rewrite your function as a composite function.");
      aliased_input_idx = i;
      const c10::IValue& aliased_input_iv =
          (*stack)[stack_start + i]; // get a reference to an ivalue on the
                                     // stack
      TORCH_CHECK(aliased_input_iv.isTensor());
      aliased_input =
          aliased_input_iv.toTensor(); // TODO: Can we avoid saving this tensor
                                       // and incurring the refcount bump?
    }
  }
  // See NOTE [ Limitations of ADInplaceOrView boxed kernel ] above
  TORCH_CHECK(
      (aliased_input_idx == -1 && aliased_output_idx == -1) ||
          (aliased_input_idx == 0 && aliased_output_idx == 0),
      "Fallback ADInplaceOrView kernel can only create view relationships between the first "
      "input and the first output (the output can be a vector of tensors). Please change the "
      "order of your operator's parameters so that this is the case.");
  const bool is_view = aliased_input_idx != -1;

  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatchBoxed(
        dispatch_keys & c10::after_ADInplaceOrView_keyset, stack);
  }

  for (const auto i : c10::irange(num_returns)) {
    if (schema.is_mutable({c10::SchemaArgType::output, i})) {
      increment_version((*stack)[stack->size() - num_returns + i].toTensor());
    }
  }

  if (is_view) {
    c10::IValue& aliased_output_iv =
        (*stack)[stack->size() - num_returns + aliased_output_idx];
    if (aliased_output_iv.isTensorList()) {
      auto aliased_output = aliased_output_iv.toTensorVector();
      // Only allow rebasing of the history if we return a single Tensor that is
      // why we don't have to care about the view_func logic below.
      // See NOTE [ View + Inplace detection ] for more details about this logic
      auto result = as_view(
          /* base=*/aliased_input,
          /* tensors=*/aliased_output,
          /* is_bw_differentiable=*/true,
          /* is_fw_differentiable=*/true,
          /* creation_meta=*/
          InferenceMode::is_enabled()
              ? CreationMeta::INFERENCE_MODE
              : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE
                                            : CreationMeta::NO_GRAD_MODE));
      // ^ pass in creation meta unecessarily even if not isDifferentiableType,
      // but we don't have that
      //   information here anyway.
      stack->at(stack->size() - num_returns + aliased_output_idx) = result;
    } else {
      TORCH_CHECK(aliased_output_iv.isTensor());
      auto result = as_view(
          /* base=*/aliased_input,
          /* tensor=*/std::move(aliased_output_iv).toTensor(),
          /* is_bw_differentiable=*/true,
          /* is_fw_differentiable=*/true,
          /* view_func=*/
          [op_name = op_name](const at::Tensor&) {
            // We always need this view_func because otherwise if we do in-place
            // on this view, we would implicitly use AsStridedBackward instead
            // of the NotImplemented node. For the cross-dtype/non-strided
            // cases, we would create something like this anyway
            TORCH_CHECK(
                false,
                "Mutating the view ",
                op_name,
                " which does not have a derivative implemented is forbidden.");
            return at::Tensor();
          },
          /* creation_meta=*/
          InferenceMode::is_enabled()
              ? CreationMeta::INFERENCE_MODE
              : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT
                                            : CreationMeta::NO_GRAD_MODE));
      stack->at(stack->size() - num_returns + aliased_output_idx) = result;
    }
  }
}

torch::CppFunction autogradNotImplementedInplaceOrViewFallback() {
  return torch::CppFunction::makeFromBoxedFunction<
      &autogradNotImplementedInplaceOrViewFallbackImpl>();
}

} // namespace autograd
} // namespace torch
