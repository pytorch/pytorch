#pragma once

#include <torch/custom_class.h>

namespace torch {
namespace jit {

// Interface for a JIT backend.
class TORCH_API PyTorchBackendInterface : public torch::CustomClassHolder {
 public:
  PyTorchBackendInterface() noexcept;
  ~PyTorchBackendInterface() override;

  // Returns true if the backend is available to process delegation calls.
  virtual bool is_available() = 0;

  // Compile the module contained in \p processed using the details provided in
  // \p method_compile_spec for each module method that should be compiled for
  // the backend. \p method_compile_spec should be of type Dict<string, Any>.
  // \returns a dictionary of type Dict<string, Any> that contains a backend
  // handle each method that can run on the backend (i.e. each key in \p
  // method_compile_spec).
  virtual c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) = 0;

  // Execute the method specified by \p handle using \p inputs. \returns the
  // outputs as a tuple.
  virtual c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) = 0;
};
} // namespace jit
} // namespace torch
