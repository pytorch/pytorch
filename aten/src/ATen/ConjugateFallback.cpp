#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>

namespace at {

void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // Situations to handle:
  //  1. Purely functional situation.  Easy: materialize all inputs and
  //     call it a day.
  //  2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
  //     Materialize other inputs as in (1).
  //  3. Out-of-place operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
  //     Materialize other inputs as in (1).
  //
  // It is important to be able to tell if we READ from an argument and if we
  // WRITE from an argument.  Conservative approach is to assume that we always
  // READ from an argument, but in out-of-place operations you can skip
  // conjugating inputs on entry that never get used.  In current schema we
  // can't easily tell if inplace situation has happened, so don't do it.

  // std::cerr << "conj fallback " << op.schema().name() << "\n";

  const auto& arguments = op.schema().arguments();
  const auto num_arguments = arguments.size();
  const auto stack_start = stack->size() - num_arguments;

  // Mutable inputs to be tracked separately
  std::vector<Tensor> mutable_inputs;

  for (int64_t i = 0; i < num_arguments; ++i) {
    auto& ivalue = (*stack)[stack_start + i];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& argument = arguments[i];
    bool is_write = false;
    if (argument.alias_info()) {
      TORCH_CHECK(argument.alias_info()->isWrite(),
        "Conjugate fallback doesn't work for non-mutable input arguments that "
        "alias with outputs, these must be implemented manually.  "
        "If you got this error on a core op, please report a bug to PyTorch");
      is_write = true;
    }
    auto* impl = ivalue.unsafeToTensorImpl();
    if (!impl->is_conj()) {
      continue;
    }

    auto tensor = std::move(ivalue).toTensor();
    if (is_write) {
      // TODO: This is a waste if the argument is write only
      native::conj_physical_(tensor);
      tensor.set_conj(false);
      mutable_inputs.emplace_back(tensor);
    } else {
      tensor = native::resolve_conj(tensor);
    }
    (*stack)[stack_start + i] = std::move(tensor);
  }

  op.callBoxed(stack);

  for (auto& mutable_input : mutable_inputs) {
    native::conj_physical_(mutable_input);
    mutable_input.set_conj(true);
  }
}

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("conj", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
  m.impl("empty_like", torch::CppFunction::makeFallthrough());
  m.impl("empty.out", torch::CppFunction::makeFallthrough());
  m.impl("empty_strided", torch::CppFunction::makeFallthrough());
  m.impl("stride.int", torch::CppFunction::makeFallthrough());
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("is_complex", torch::CppFunction::makeFallthrough());
  m.impl("view_as_real_physical", torch::CppFunction::makeFallthrough());
  m.impl("view_as_real", torch::CppFunction::makeFallthrough());
  m.impl("imag", torch::CppFunction::makeFallthrough());
  m.impl("real", torch::CppFunction::makeFallthrough());
  m.impl("view", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
  // TODO: need to hit the view functions
}

} // namespace at
