#include <torch/csrc/jit/backends/backend_detail.h>

#include <ATen/core/builtin_function.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace detail {
c10::FunctionSchema getPreprocessSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("mod", c10::AnyType::get());
  c10::Argument method_compile_spec(
      "method_compile_spec",
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get()));

  c10::FunctionSchema preprocessor_schema(
      "preprocess",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{mod});
  return preprocessor_schema;
}

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

std::unordered_map<std::string, BackendPreprocessFunction>&
backendPreprocessFunctions() {
  static std::unordered_map<std::string, BackendPreprocessFunction>
      preprocess_functions;
  return preprocess_functions;
}

bool hasBackendPreprocessFunction(const std::string& name) {
  return backendPreprocessFunctions().count(name);
}

void registerBackendPreprocessFunction(
    const std::string& name,
    const BackendPreprocessFunction& preprocess) {
  TORCH_CHECK(
      !detail::hasBackendPreprocessFunction(name),
      "BackendPreprocessFunction for backend ",
      name,
      " is already registered. Ensure that registration is only called once.");
  detail::backendPreprocessFunctions()[name] = preprocess;
}

BackendPreprocessFunction getBackendPreprocessFunction(
    const std::string& name) {
  TORCH_CHECK(
      hasBackendPreprocessFunction(name),
      "BackendPreprocessFunction for backend ",
      name,
      " is not registered.");
  return backendPreprocessFunctions()[name];
}
} // namespace detail
} // namespace jit
} // namespace torch
