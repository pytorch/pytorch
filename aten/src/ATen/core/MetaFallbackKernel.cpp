#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/PyInterpreter.h>
#include <torch/library.h>

namespace at::impl {

static void metaFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  c10::Dispatcher::singleton().throwIfHasAbstractImplPyStub(op.operator_name());
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      op.operator_name(),
      ": attempted to run this operator with Meta tensors, but there was no ",
      "abstract impl or Meta kernel registered. You may have run into this message "
      "while using an operator with PT2 compilation APIs (torch.compile/torch.export); "
      "in order to use this operator with those APIs you'll need to add an abstract impl."
      "Please see the following doc for next steps: "
      "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit");
}

TORCH_LIBRARY_IMPL(_, Meta, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&metaFallback>());
}

} // namespace at::impl
