#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace custom_backend {
// This custom JIT backend is intended to do the minimal amount of work
// necessary to test that the JIT backend registration endpoints and
// code generation are working correctly. It is not intended to
// produce numerically correct results.
class CustomBackend : public torch::jit::PyTorchBackendInterface {
 public:
  // Constructor.
  explicit CustomBackend() {}
  virtual ~CustomBackend() = default;

  bool is_available() override {
    return true;
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto spec =
        c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

    // Return the same string as a value for every key in method_compile_spec.
    auto handles = c10::Dict<std::string, std::string>();
    for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
      handles.insert(it->key(), it->key());
    }
    return c10::impl::toGenericDict(handles);
  }
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(handle.isString());
    TORCH_INTERNAL_ASSERT(inputs.size() > 0);

    c10::List<at::Tensor> output_list;

    // Implement simple accumulator and negative accumulator (?) ops. Return one
    // or both of them depending on the handle to make sure multiple outputs are
    // handled.
    c10::IValue value = inputs[0];
    at::Tensor accum = value.toTensor();
    accum = accum.clone();
    at::Tensor sub_accum = value.toTensor();
    sub_accum = sub_accum.clone();

    for (size_t i = 1, e = inputs.size(); i < e; ++i) {
      value = inputs[i];
      accum.add_(value.toTensor(), 1.0);
      sub_accum.sub_(value.toTensor(), 1.0);
    }

    if (handle.toStringRef() == "accum") {
      output_list.emplace_back(accum);
    } else if (handle.toStringRef() == "sub_accum") {
      output_list.emplace_back(sub_accum);
    } else if (handle.toStringRef() == "forward") {
      output_list.emplace_back(accum);
      output_list.emplace_back(sub_accum);
    }

    return c10::impl::toList(output_list);
  }
};

c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec) {
  return mod._ivalue();
}

// clang-format off
#  if defined(_WIN32)
#    if defined(custom_ops_EXPORTS)
#      define CUSTOM_BACKEND_API __declspec(dllexport)
#    else
#      define CUSTOM_BACKEND_API __declspec(dllimport)
#    endif
#  else
#    define CUSTOM_BACKEND_API
#  endif
// clang-format on

CUSTOM_BACKEND_API std::string getBackendName();
} // namespace custom_backend
} // namespace torch
