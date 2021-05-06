#pragma once

#include <ATen/core/builtin_function.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/backends/backend_interface.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {
namespace {
// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
c10::FunctionSchema getIsAvailableSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument available("available", c10::BoolType::get());
  c10::FunctionSchema preprocessor_schema(
      "is_available",
      /*overload_name=*/"",
      /*arguments=*/{self},
      /*returns=*/{available});
  return preprocessor_schema;
}

constexpr static auto kBackendsNamespace = "__backends__";

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
c10::FunctionSchema getCompileSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("processed", c10::AnyType::get());
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  c10::Argument method_compile_spec("method_compile_spec", any_dict_ty);
  c10::Argument handles("handles", any_dict_ty);

  c10::FunctionSchema compile_schema(
      "compile",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{handles});
  return compile_schema;
}

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
c10::FunctionSchema getExecuteSchema() {
  auto any_list_ty = c10::ListType::create(c10::AnyType::get());
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument handle("handle", c10::AnyType::get());
  c10::Argument input("input", any_list_ty);
  c10::Argument output("output", any_list_ty);
  return c10::FunctionSchema(
      "execute",
      /*overload_name=*/"",
      /*arguments=*/{self, handle, input},
      /*returns=*/{output});
}

template <typename TBackendInterface>
std::function<void(Stack&)> getIsAvailableFunc() {
  return [](Stack& stack) {
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->is_available();
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
} // namespace

// Static registration API for backends.
template <class TBackendInterface>
class backend {
  static_assert(
      std::is_base_of<PyTorchBackendInterface, TBackendInterface>::value,
      "torch::jit::backend<T> requires T to inherit from PyTorchBackendInterface");
  std::string backend_name_;

 public:
  // Registers a new backend with /p name, and the given /p preprocess
  // function.
  backend(const std::string& name) : backend_name_(name) {
    static auto cls = torch::class_<TBackendInterface>(kBackendsNamespace, name)
                          .def(torch::init<>())
                          ._def_unboxed(
                              "is_available",
                              getIsAvailableFunc<TBackendInterface>(),
                              getIsAvailableSchema())
                          ._def_unboxed(
                              "compile",
                              getCompileFunc<TBackendInterface>(),
                              getCompileSchema())
                          ._def_unboxed(
                              "execute",
                              getExecuteFunc<TBackendInterface>(),
                              getExecuteSchema());
  }
};

} // namespace jit
} // namespace torch
