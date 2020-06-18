#include <torch/csrc/jit/backends/backend_detail.h>
#include <ATen/core/builtin_function.h>

namespace torch {
namespace jit {
namespace detail {

namespace {
// Get a reference to the registry of all currently registered backends. The
// primary use for this is to create Python bindings for lowering a Module to
// every backend in the registry.
std::vector<std::string>& getBackendRegistry() {
  static std::vector<std::string> registry;
  return registry;
}

// Get a reference to the list of all callbacks that should be called when a
// backend is registered. This is primarily used for creating Python bindings
// for lowering to backends from Python.
std::vector<BackendRegistrationCallback>& getBackendRegistrationCallbacks() {
  static std::vector<BackendRegistrationCallback> callbacks;
  return callbacks;
}
} // namespace

void registerBackend(const std::string& backend_name) {
  // Call all registered callbacks on this backend.
  for (auto& callback : getBackendRegistrationCallbacks()) {
    callback(backend_name);
  }

  // Add the backend to the backend registry.
  getBackendRegistry().emplace_back(backend_name);
}

void addBackendRegistrationCallback(BackendRegistrationCallback callback) {
  // Call this callback on all registered backends.
  for (auto& backend : getBackendRegistry()) {
    callback(backend);
  }

  // Add the callback to the callback registry.
  getBackendRegistrationCallbacks().emplace_back(std::move(callback));
}

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

} // namespace detail
} // namespace jit
} // namespace torch
