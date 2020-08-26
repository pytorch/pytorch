#pragma once

#include <ATen/core/stack.h>

namespace torch {
namespace jit {
namespace detail {

constexpr static auto kBackendsNamespace = "__backends__";

c10::FunctionSchema TORCH_API getPreprocessSchema();
c10::FunctionSchema TORCH_API getCompileSchema();
c10::FunctionSchema TORCH_API getExecuteSchema();

template <typename TBackendInterface>
std::function<void(Stack&)> getPreprocessFunc() {
  return [](Stack& stack) {
    auto method_compile_spec = pop(stack).toGenericDict();
    auto mod = pop(stack);
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->preprocess(mod, method_compile_spec);
    push(stack, ret);
  };
}

template <typename TBackendInterface>
std::function<void(Stack&)> getCompileFunc() {
  return [](Stack& stack) {
    auto method_compile_spec = pop(stack).toGenericDict();
    auto processed = pop(stack);
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->compile(processed, method_compile_spec);
    push(stack, ret);
  };
}

template <typename TBackendInterface>
std::function<void(Stack&)> getExecuteFunc() {
  return [](Stack& stack) {
    auto args = pop(stack);
    auto handle = pop(stack);
    auto self = pop(stack);
    auto backend = self.toCustomClass<TBackendInterface>();
    auto res = backend->execute(handle, args.toList());
    push(stack, res);
  };
}
} // namespace detail
} // namespace jit
} // namespace torch
