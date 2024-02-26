#pragma once

#include <c10/macros/Export.h>

#include <cstdint>
#include <memory>

namespace c10 {

enum class C10_API_ENUM DebugInfoKind : uint8_t {
  PRODUCER_INFO = 0,
  MOBILE_RUNTIME_INFO,
  PROFILER_STATE,
  INFERENCE_CONTEXT, // for inference usage
  PARAM_COMMS_INFO,

  TEST_INFO, // used only in tests
  TEST_INFO_2, // used only in tests
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

 private:
  bool active_ = false;
  std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

} // namespace c10
