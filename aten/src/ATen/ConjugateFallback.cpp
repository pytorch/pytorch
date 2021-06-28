#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>

namespace at {

void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // Situations to handle:
  //  1. Out-of-place operation.  Easy: materialize all inputs and
  //     call it a day.
  //  2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
  //     Materialize other inputs as in (1).
  //  3. out= operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
  //  Materialize other inputs as in (1).
  //
  //  It is important to be able to tell if we READ from an argument and if we
  //  WRITE from an argument.  Conservative approach is to assume that we always
  //  READ from an argument, but in out-of-place operations you can skip
  //  conjugating inputs on entry that never get used.  In current schema we
  //  can't easily tell if inplace situation has happened, so don't do it.

  const auto& arguments = op.schema().arguments();
  const auto num_arguments = arguments.size();
  const auto stack_start = stack->size() - num_arguments;

  c10::optional<bool> is_write;
  for (int64_t i = 0; i < num_arguments; ++i) {
    const auto& alias_info = arguments[i].alias_info();
    // Three possible states:
    // 1. alias_info has no value --> out-of-place operation
    // 2. alias_info does have a value, alias_info->is_write=True --> in-place or out= operation
    // 3. alias_info does have a value, alias_info->is_write=False --> view operation
    if (alias_info.has_value()) {
      if (is_write.has_value()) {
        TORCH_CHECK(*is_write == alias_info->isWrite(),
          "Unsupported operator for conjugate fallback: ", op.schema().name(),
          "Conjugate fallback doesn't work for operators with a mix "
          "mutable and non-mutable inputs that alias with outputs, "
           "this must be implemented manually.  "
          "If you got this error on a core op, please report a bug to PyTorch.");
      } else {
        is_write = alias_info->isWrite();
      }
    }
  }

  if (is_write.has_value() && !*is_write) {
    // We assume that view operators automatically handle conjugation
    // correctly by propagating the Conjugate dispatch key in key_set.
    // This is not necessarily always right, so you should test these cases.
    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::Conjugate), stack);
    return;
  }

  // Mutable inputs to be tracked separately
  std::vector<Tensor> mutable_inputs;

  for (int64_t i = 0; i < num_arguments; ++i) {
    auto& ivalue = (*stack)[stack_start + i];
    if (!(ivalue.isTensor() || ivalue.isTensorList())) {
      continue;
    }
    const auto& argument = arguments[i];
    bool mut_arg = false;
    if (argument.alias_info()) {
      // View operations were already filtered above, so only in-place/out= operations should get here.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
      mut_arg = true;
    }
    if (ivalue.isTensor()) {
      auto* impl = ivalue.unsafeToTensorImpl();
      if (!impl->is_conj()) {
        continue;
      }

      auto tensor = std::move(ivalue).toTensor();
      TORCH_CHECK_NOT_IMPLEMENTED(!tensor.is_meta(), "Conjugate Fallback does not support meta tensors.");
      if (mut_arg) {
        // TODO: This is a waste if the argument is write only
        tensor._set_conj(false);
        at::conj_physical_(tensor);
        mutable_inputs.emplace_back(tensor);
      } else {
        tensor = at::resolve_conj(tensor);
      }
      (*stack)[stack_start + i] = std::move(tensor);
    } else if (ivalue.isTensorList()) {
      auto tensors = std::move(ivalue).toTensorList();
      if (mut_arg) {
        for(const auto j : c10::irange(tensors.size())) {
          Tensor t = tensors[j];
          t._set_conj(false);
          at::conj_physical_(t);
          mutable_inputs.emplace_back(t);
        }
      } else {
        for(const auto j : c10::irange(tensors.size())) {
          tensors[j] = at::resolve_conj(tensors[j]);
        }
      }
      (*stack)[stack_start + i] = std::move(tensors);
    }
  }


  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::Conjugate), stack);

    for (auto& mutable_input : mutable_inputs) {
      at::conj_physical_(mutable_input);
      mutable_input._set_conj(true);
    }
}

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
  m.impl("empty_like", torch::CppFunction::makeFallthrough());
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough());
  m.impl("empty.out", torch::CppFunction::makeFallthrough());
  m.impl("empty_strided", torch::CppFunction::makeFallthrough());
  m.impl("full_like", torch::CppFunction::makeFallthrough());
  m.impl("stride.int", torch::CppFunction::makeFallthrough());
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("size.int", torch::CppFunction::makeFallthrough());
  m.impl("size.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("is_complex", torch::CppFunction::makeFallthrough());
  m.impl("_view_as_real_physical", torch::CppFunction::makeFallthrough());
  m.impl("view_as_real", torch::CppFunction::makeFallthrough());
  m.impl("imag", torch::CppFunction::makeFallthrough());
  m.impl("real", torch::CppFunction::makeFallthrough());
  m.impl("view", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
}

} // namespace at
