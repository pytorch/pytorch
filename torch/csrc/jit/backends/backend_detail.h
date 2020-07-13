#pragma once

#include <ATen/core/stack.h>

namespace torch {
namespace jit {
namespace detail {

constexpr static auto kBackendsNamespace = "__backends__";

using BackendRegistrationCallback = std::function<void(const std::string&)>;

// Add a function \p callback that should be invoked every time a backend is
// registered. The name of the backend is passed to the callback. This is
// primarily used for creating Python bindings for lowering to backends from
// Python.
void addBackendRegistrationCallback(BackendRegistrationCallback callback);

// Register the provided \p lowering_function in the backend registry
// under the name \p backend_name.
void registerBackend(const std::string& backend_name);

c10::FunctionSchema getPreprocessSchema();
c10::FunctionSchema getCompileSchema();
c10::FunctionSchema getExecuteSchema();

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
