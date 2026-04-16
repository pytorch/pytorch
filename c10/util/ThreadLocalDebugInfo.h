#pragma once

#include <c10/macros/Export.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string_view>

namespace c10 {

// Identifies a slot in ThreadLocalDebugInfo for storing a specific type
// of python-thread-local state.
//
// This class is trivial to copy and move, and is cheap to construct and
// destroy. It mimics the behavior of an enum class for backward compatibility
// with existing uses.
class C10_API DebugInfoKind {
 private:
  // Must be a non-null pointer to a string literal with static storage
  // duration. The pointer address is used as the identifier for the slot.
  // The string itself is only used for debugging purposes and should be
  // descriptive of the slot's purpose and ideally (but not required) be unique.
  using value_type = const std::string_view*;  // Must be non-null.

 public:
  // Creates an uninitialized DebugInfoKind.
  DebugInfoKind() = default;

  // Creates a DebugInfoKind with the given identity.
  explicit DebugInfoKind(value_type value);

  // Predefined DebugInfoKinds for common use cases.
  static const DebugInfoKind PRODUCER_INFO;
  static const DebugInfoKind MOBILE_RUNTIME_INFO;
  static const DebugInfoKind PROFILER_STATE;
  static const DebugInfoKind INFERENCE_CONTEXT;  // for inference usage
  static const DebugInfoKind PARAM_COMMS_INFO;
  static const DebugInfoKind TEST_INFO;    // used only in tests
  static const DebugInfoKind TEST_INFO_2;  // used only in tests

  constexpr bool operator==(const DebugInfoKind& other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(const DebugInfoKind& other) const {
    return value_ != other.value_;
  }

  friend std::ostream& operator<<(std::ostream& os, const DebugInfoKind& kind);

 private:
  value_type value_ = nullptr;
};

class C10_API DebugInfoBase {
 public:
  DebugInfoBase() = default;
  virtual ~DebugInfoBase() = default;
};

// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is supposed
// to be utilized by the user's code to pass extra information from
// the higher layers (e.g. model id) down to the lower levels
// (e.g. to the operator observers used for debugging, logging,
// profiling, etc)
class C10_API ThreadLocalDebugInfo {
 public:
  static DebugInfoBase* get(DebugInfoKind kind);

  // Get current ThreadLocalDebugInfo
  static std::shared_ptr<ThreadLocalDebugInfo> current();

  // Internal, use DebugInfoGuard/ThreadLocalStateGuard
  static void _forceCurrentDebugInfo(
      std::shared_ptr<ThreadLocalDebugInfo> info);

  // Push debug info struct of a given kind
  static void _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);
  // Pop debug info, throws in case the last pushed
  // debug info is not of a given kind
  static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);
  // Peek debug info, throws in case the last pushed debug info is not of the
  // given kind
  static std::shared_ptr<DebugInfoBase> _peek(DebugInfoKind kind);

 private:
  std::shared_ptr<DebugInfoBase> info_;
  DebugInfoKind kind_;
  std::shared_ptr<ThreadLocalDebugInfo> parent_info_;

  friend class DebugInfoGuard;
};

// DebugInfoGuard is used to set debug information,
// ThreadLocalDebugInfo is semantically immutable, the values are set
// through the scope-based guard object.
// Nested DebugInfoGuard adds/overrides existing values in the scope,
// restoring the original values after exiting the scope.
// Users can access the values through the ThreadLocalDebugInfo::get() call;
class C10_API DebugInfoGuard {
 public:
  DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

  explicit DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info);

  ~DebugInfoGuard();

  DebugInfoGuard(const DebugInfoGuard&) = delete;
  DebugInfoGuard(DebugInfoGuard&&) = delete;
  DebugInfoGuard& operator=(const DebugInfoGuard&) = delete;
  DebugInfoGuard& operator=(DebugInfoGuard&&) = delete;

 private:
  bool active_ = false;
  std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

} // namespace c10
