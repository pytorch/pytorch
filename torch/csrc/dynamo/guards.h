#pragma once
#include <c10/core/GradMode.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::dynamo {

PyObject* torch_c_dynamo_guards_init();

// interfaces for extra_state and eval_frame.c because RootGuardManager class is
// not visible there.
void* convert_to_root_guard_manager(py::object root);
bool run_root_guard_manager(void* root, FrameLocalsMapping* f_locals);

extern thread_local bool tls_is_in_mode_without_ignore_compile_internals;

void set_is_in_mode_without_ignore_compile_internals(bool value);

// If we're in a mode with ignore_compile_internals=False, we WON'T mask
// Python keys from guard checking (they should be visible, so eager fallback is
// possible). Otherwise (invisible mode or no mode), we WILL mask Python keys to
// avoid guard failures on the dispatch keyset at runtime.
bool get_is_in_mode_without_ignore_compile_internals();

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  c10::DispatchKeySet override_dispatch_key_set;
  bool grad_mode_enabled;
  bool should_mask_python_keys;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    if (override_dispatch_key_set.empty()) {
      auto result =
          (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;

      if (should_mask_python_keys) {
        result = result -
            c10::DispatchKeySet(
                     {c10::DispatchKey::Python,
                      c10::DispatchKey::PythonTLSSnapshot,
                      c10::DispatchKey::PythonDispatcher});
      }

      return result;
    } else {
      return override_dispatch_key_set;
    }
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        override_dispatch_key_set(c10::BackendComponent::InvalidBit),
        grad_mode_enabled(at::GradMode::is_enabled()),
        should_mask_python_keys(
            !get_is_in_mode_without_ignore_compile_internals()) {}

  void overrideDispatchKeySet(c10::DispatchKeySet ks) {
    override_dispatch_key_set = ks;
  }
};

class TensorCheck {
 public:
  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      const at::Tensor& v,
      c10::DispatchKeySet dispatch_key_set,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_strides);

  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      c10::DispatchKeySet dispatch_key_set,
      at::ScalarType dtype,
      at::DeviceIndex device_index,
      bool requires_grad,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_strides);

  bool check(const LocalState& state, const at::Tensor& v);
  bool check(
      const LocalState& state,
      const c10::DispatchKeySet& dispatch_key_set,
      const at::ScalarType& dtype,
      const c10::Device& device,
      const c10::SymIntArrayRef& dynamic_dims_sizes,
      const c10::SymIntArrayRef& dynamic_dims_strides,
      const bool& requires_grad);
  std::string check_verbose(
      const LocalState& state,
      const at::Tensor& v,
      const std::string& tensor_name);

  PyTypeObject* pytype;

 private:
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  at::ScalarType dtype_;
  // Note(voz): While dispatch_key_ is sufficiently representative of a device
  // In that keys are more granular AND device specific - they do not
  // necessarily capture device indices correctly.
  at::DeviceIndex device_index_;
  bool requires_grad_;
  // NB: These are unset if dynamic shapes is enabled.
  std::vector<std::optional<c10::SymInt>> sizes_;
  std::vector<std::optional<c10::SymInt>> strides_;
  // Not strictly required for dense tensors, but nested tensors need it.
  int64_t dim_;
};

} // namespace torch::dynamo
