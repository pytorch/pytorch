#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/core/StorageImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace c10 {

struct TensorImpl;

// State shared by C++ fake tensors and Python mode wrappers.
// no in_kernel_invocation flag; that state is thread-local in C++
// (see Note [C++ in_kernel_invocation] in FakeTensorModeTLS.h)
struct C10_API FakeTensorMode {
  std::shared_ptr<c10::SafePyObject> shape_env_;
  std::shared_ptr<c10::SafePyObject> fake_tensor_converter_;
  // CapturedTraceback keeps code locations, not frames or their fake tensors.
  // Owned here so a recreated wrapper still reports the original creation.
  std::shared_ptr<c10::SafePyObject> creation_traceback_;
  // Owns a weakref to the Python wrapper, which can be recreated while this
  // mode is alive. Accessed under fake_mode_pyobj_mutex_; see
  // Note [C++ FakeTensorMode Python object lifetime].
  std::shared_ptr<c10::SafePyObject> fake_mode_pyobj_;
  std::mutex fake_mode_pyobj_mutex_;

  // The fields below are plain, like a Python FakeTensorMode's attributes, and
  // must not be changed while another thread is using the mode.

  // when false, disallow a fake tensor from having a 'meta' device
  bool allow_meta_ = true;

  // Snapshot of fake_tensor_prefer_device_type for native device resolution.
  // We assume the config stays fixed for the mode's lifetime; we have not found
  // real use cases requiring changes during that lifetime. Python op_impls
  // still read live config, so changing it can give different device decisions
  // across paths; see Note [C++ fake fallback]. Kept as a string to preserve
  // Python's substring matching of device types.
  std::optional<std::string> prefer_device_type_;

  // When true, real tensor inputs to an op are converted to fakes instead of
  // raising. A non-None fake_tensor_tls.allow_non_fake_inputs_override, a
  // thread-local override used by dynamo's nonstrict_trace, takes precedence,
  // so read this through PyInterpreterVTable::allow_non_fake_inputs().
  bool allow_non_fake_inputs_ = false;

  // Default used by from_tensor when its static_shapes argument is None.
  bool static_shapes_ = shape_env_ == nullptr;

  // Whether reading the mutable data pointer of a fake made afterwards only
  // warns (true) or throws (false); a storage's marking is never relaxed.
  // Python's _allow_unsafe_data_ptr_access, exposed under that name.
  bool allow_unsafe_data_ptr_access_ = true;

  // The mode's epoch (FakeTensorMode.epoch), which code that retraces bumps. A
  // fake's memos are kept only for the epoch they were set in, unless kept
  // across epochs; the memo bindings read it through the fake's mode.
  uint64_t epoch_ = 0;

  // Whether ops also run on the real tensors the fakes carry.
  bool propagate_real_tensors_ = false;

  // Defined out of line, as both instantiate constant_storage_mapping_'s
  // destructor, which needs TensorImpl to be complete.
  FakeTensorMode(
      std::shared_ptr<c10::SafePyObject> shape_env,
      std::shared_ptr<c10::SafePyObject> converter,
      bool allow_meta = true,
      bool allow_unsafe_data_ptr_access = true,
      bool propagate_real_tensors = false);
  ~FakeTensorMode();

  // record the real constant a fake tensor was created from; the constant is
  // stored on the fake's ExtraMeta so it dies with the tensor
  // setting to nullptr to clear it
  void set_constant(
      c10::TensorImpl* fake_impl,
      c10::intrusive_ptr<c10::TensorImpl> constant);
  void clear_constant(c10::TensorImpl* fake_impl) noexcept;

  // return the real constant a fake tensor was created from, or nullptr
  const c10::intrusive_ptr<c10::TensorImpl>& get_constant(
      c10::TensorImpl* fake_impl) const;
  // drop constant tracking for fake tensors aliasing this mutated storage
  void invalidate_constant_aliases(c10::StorageImpl* storage_impl);
  // drop non-CPU constants, keeping cheap CPU ones for constant folding
  void clear_non_cpu_constants();

 private:
  // key = constant storage, values = all fake tensors that share this storage
  // (aliases)
  struct ConstantAliases {
    c10::weak_intrusive_ptr<c10::StorageImpl> storage;
    std::vector<c10::weak_intrusive_ptr<c10::TensorImpl>> tensors;
  };
  std::unordered_map<c10::StorageImpl*, ConstantAliases>
      constant_storage_mapping_;
};

} // namespace c10
