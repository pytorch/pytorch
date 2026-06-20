#pragma once
#include <c10/core/GradMode.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::dynamo {

PyObject* torch_c_dynamo_guards_init();

// interfaces for extra_state and eval_frame.c because RootGuardManager class is
// not visible there.
void* convert_to_root_guard_manager(py::object root);
bool run_root_guard_manager(void* root, FrameLocalsMapping* f_locals);

struct GuardLastSuccessReceipt;
bool run_root_guard_manager(
    void* root,
    FrameLocalsMapping* f_locals,
    GuardLastSuccessReceipt* receipt);
bool run_root_guard_manager_with_last_success_receipt(
    GuardLastSuccessReceipt* receipt,
    void* entry_key,
    void* root,
    FrameLocalsMapping* f_locals,
    bool is_skip_guard_eval_unsafe);

enum class GuardPartialMemoState : uint8_t {
  Training = 0,
  Enabled = 1,
  Disabled = 2,
};

enum class GuardSubtreeTokenKind : uint8_t {
  ObjectIdentity,
  ObjectType,
  DictVersion,
  SequenceSize,
  TensorMetadata,
};

enum class ActualPartialReplayFailureReason : uint8_t {
  None = 0,
  UnsupportedAccessor = 1,
  UnsupportedLeaf = 2,
};

struct GuardSubtreeEntryToken {
  GuardSubtreeTokenKind kind;
  uintptr_t object_id = 0;
  uintptr_t type_id = 0;
  uint64_t version = 0;
  int64_t size = -1;
};

inline bool operator==(
    const GuardSubtreeEntryToken& lhs,
    const GuardSubtreeEntryToken& rhs) {
  return lhs.kind == rhs.kind && lhs.object_id == rhs.object_id &&
      lhs.type_id == rhs.type_id && lhs.version == rhs.version &&
      lhs.size == rhs.size;
}

struct GuardLastSuccessReceipt {
  void* actual_partial_entry_key = nullptr;
  void* actual_partial_root_key = nullptr;
  PyObject* actual_partial_self_object = nullptr;
  PyTypeObject* actual_partial_self_type = nullptr;
  uint64_t actual_partial_shadow_passes = 0;
  uint64_t actual_partial_hit = 0;
  uint64_t actual_partial_miss = 0;
  uint64_t slow_guard_fallback = 0;
  std::unordered_map<std::string, uint64_t> actual_partial_disabled_reasons;
  GuardPartialMemoState actual_partial_state = GuardPartialMemoState::Training;
  std::vector<GuardSubtreeEntryToken> actual_partial_stability_tokens;
  std::vector<GuardSubtreeEntryToken> actual_partial_tokens;
};

std::unique_ptr<GuardLastSuccessReceipt> create_guard_last_success_receipt();

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  c10::DispatchKeySet override_dispatch_key_set;
  bool grad_mode_enabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    if (override_dispatch_key_set.empty()) {
      return (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;
    } else {
      return override_dispatch_key_set;
    }
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        override_dispatch_key_set(c10::BackendComponent::InvalidBit),
        grad_mode_enabled(at::GradMode::is_enabled()) {}

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
