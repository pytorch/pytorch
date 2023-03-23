#include <ATen/view/TransformFallback.h>

#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at::view {

TransformFallback::TransformFallback(c10::DispatchKey key) : key_(key) {}
TransformFallback::~TransformFallback() = default;

auto TransformFallback::operator()(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, c10::Stack* stack) -> void {
  /*
    Situations to handle:
    1. Out-of-place operation.  Easy: materialize all inputs and
    call it a day.
    2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
    Materialize other inputs as in (1).
    3. out= operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
    Materialize other inputs as in (1).

    It is important to be able to tell if we READ from an argument and if we
    WRITE to an argument.  Conservative approach is to assume that we always
    READ from an argument, but in out= operations you can skip
    conjugating inputs on entry that never get used. In the current schema we
    can't easily tell if the operation is in in-place or out= operation.

    Note:
    1. Mutable tensorlists containing tensors whose math bit set to true are disallowed.
    2. Mutable tensors with math bit set to true are unconditionally cloned to ensure
    correct behavior in the case when the mutable tensor shares memory with non mutable arguments.

    If we were to in-place resolve the math bit for mutable inputs, then the non-mutable inputs sharing partial or full memory
    with these mutable inputs would read into wrong values in the following cases:
    1. Non mutable inputs have their math bit set to false.
    2. Math bit for mutable input(s) is resolved before the non mutable inputs (with bit set to true and sharing memory
    with one or more mutable arg(s)) are cloned.
    At the end, the final value of the mutable arguments from the stack are copied into the original input mutable tensor inputs.
  */
  const auto& arguments = op.schema().arguments();
  const auto num_arguments = arguments.size();
  const auto stack_start = stack->size() - num_arguments;

  c10::optional<bool> is_write;
  for (const auto i : c10::irange(num_arguments)) {
    // Three possible states:
    // 1. alias_info has no value --> out-of-place operation
    // 2. alias_info does have a value, alias_info->is_write=True --> in-place or out= operation
    // 3. alias_info does have a value, alias_info->is_write=False --> view operation
    const AliasInfo* alias_info = arguments[i].alias_info();
    if (alias_info != nullptr) {
      if (is_write.has_value()) {
    TORCH_CHECK(*is_write == alias_info->isWrite(),
            "Unsupported operator for ", operator_name(), " fallback: ", op.schema().name(),
            operator_name(), " fallback doesn't work for operators with a mix "
            "mutable and non-mutable inputs that alias with outputs, "
            "this must be implemented manually.  "
            "If you got this error on a core op, please report a bug to PyTorch.");
      } else {
    is_write = alias_info->isWrite();
      }
    }
  }

  if (is_write.has_value() && !*is_write) {
    // We assume that view operators automatically handle the math bit
    // correctly by propagating the dispatch key in key_set.
    // This is not necessarily always right, so you should test these cases.
    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key_), stack);
    return;
  }

  // Mutable inputs with math bit set to True and their clones
  std::vector<std::pair<Tensor, Tensor>> mutable_inputs_with_their_clones;
  for (const auto i : c10::irange(num_arguments)) {
    auto& ivalue = (*stack)[stack_start + i];
    if (!(ivalue.isTensor() || ivalue.isTensorList())) {
      continue;
    }
    const auto& argument = arguments[i];
    bool mut_arg = false;
    if (argument.alias_info()) {
      // Was already tested by is_write loop above
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
      mut_arg = true;
    }
    if (ivalue.isTensor()) {
      if (!has_key(ivalue.toTensor())) {
    continue;
      }
      auto tensor = std::move(ivalue).toTensor();
      auto resolved_tensor = transform(tensor);
      if (mut_arg) {
    TORCH_CHECK(mutable_inputs_with_their_clones.empty(), operator_name(), " fallback does not support operators with more than one mutable tensors with ",
            operator_name(), "bit set to true.");
    mutable_inputs_with_their_clones.emplace_back(std::move(tensor), resolved_tensor);
      }
      (*stack)[stack_start + i] = std::move(resolved_tensor);
    } else if (ivalue.isTensorList()) {
      auto tensors = std::move(ivalue).toTensorList();
      for(const auto j : c10::irange(tensors.size())) {
    const auto& tensor = tensors[j];
    if (!has_key(tensor)) {
      continue;
    }
    TORCH_CHECK(!mut_arg, " fallback doesn't currently support mutable TensorLists with ",
            operator_name(), " inputs. Please materialize all the ", operator_name(), " input tensor(s) in the mutable TensorList inputs before calling ",
            op.schema().name());
    tensors[j] = transform(tensor);
      }
      (*stack)[stack_start + i] = std::move(tensors);
    }
  }

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key_), stack);

  TORCH_INTERNAL_ASSERT(mutable_inputs_with_their_clones.size() <= 1);

  for (std::pair<Tensor, Tensor> mut_tensors: mutable_inputs_with_their_clones) {
    auto& mutable_input =  mut_tensors.first;
    auto& cloned_mutable_input =  mut_tensors.second;
    auto& ivalue = (*stack)[stack_start];
    auto returned_output = std::move(ivalue).toTensor();

    // sanity check to ensure that the tensor in stack aliases the cloned_mutable_input
    TORCH_INTERNAL_ASSERT(cloned_mutable_input.is_same(returned_output));

    // necessary for out= arg
    at::native::resize_output(mutable_input, returned_output.sizes());

    untransform(mutable_input, returned_output);
    (*stack)[stack_start] = std::move(mutable_input);
  }
}

auto TransformFallback::has_key(Tensor const& tensor) const -> bool {
  return tensor.key_set().has(key_);
}

auto TransformFallback::operator_name() const -> std::string_view {
  return toString(key_);
}

} // namespace at::view
