#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <ATen/core/ivalue.h>

namespace at {

enum class DebugInfoKind : uint8_t {
  PRODUCER_INFO = 0,
  MOBILE_RUNTIME_INFO,

  TEST_INFO, // used only in tests
  TEST_INFO_2, // used only in tests
};

class CAFFE2_API DebugInfoBase {
 public:
  DebugInfoBase() {}
  virtual ~DebugInfoBase() {}
};

// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is supposed
// to be utilized by the user's code to pass extra information from
// the higher layers (e.g. model id) down to the lower levels
// (e.g. to the operator observers used for debugging, logging,
// profiling, etc)
class CAFFE2_API ThreadLocalDebugInfo {
 public:
  static std::shared_ptr<DebugInfoBase> get(DebugInfoKind kind);

  // Internal, used to propagate across thread boundaries;
  // use DebugInfoGuard instead
  static std::shared_ptr<ThreadLocalDebugInfo> _current();

  // Internal, use DebugInfoGuard/ThreadLocalStateGuard
  static void _forceCurrentDebugInfo(
      const std::shared_ptr<ThreadLocalDebugInfo>& info);

 private:
  std::shared_ptr<DebugInfoBase> debug_info_;
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
class CAFFE2_API DebugInfoGuard {
 public:
  DebugInfoGuard(
      DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

  explicit DebugInfoGuard(
      std::shared_ptr<ThreadLocalDebugInfo> info);

  ~DebugInfoGuard();

  DebugInfoGuard(const DebugInfoGuard&) = delete;
  DebugInfoGuard(DebugInfoGuard&&) = delete;

 private:
  bool active_ = false;
  std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

} // namespace at
