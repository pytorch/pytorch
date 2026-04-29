#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <iosfwd>
#include <memory>
#include <string_view>

namespace c10 {

// Identifies a slot in ThreadLocalDebugInfo for storing a specific type
// of thread-local state.
//
// This class is trivial to copy and move, and is cheap to construct and
// destroy.
//
// Example:
//
//   inline constexpr std::string_view kMyCustomInfoName = "MY_CUSTOM_INFO";
//   inline constexpr DebugInfoKind kMyCustomInfo(&kMyCustomInfoName);
//   ...
//   ThreadLocalDebugInfo::_push(kMyCustomInfo, my_custom_info_shared_ptr);
//   ...
//   const DebugInfoBase* info = ThreadLocalDebugInfo::get(kMyCustomInfo);
//   ...
//   ThreadLocalDebugInfo::_pop(kMyCustomInfo);
class DebugInfoKind {
 private:
  // Must be a non-null pointer to a string_view literal with static storage
  // duration.
  using value_type = const std::string_view*;

 public:
  // Creates an uninitialized DebugInfoKind.
  constexpr DebugInfoKind() = default;

  // Creates a DebugInfoKind with the given identity, which must be a non-null
  // pointer to a string_view literal with static storage duration. The pointer
  // address (not the string itself) is used as the identifier for the slot. The
  // string itself is only used for debugging purposes and should be descriptive
  // of the slot's purpose and ideally (but not required) be unique.
  //
  // If called with a null value in a constexpr context, will trigger a compile
  // time error. If called with a null value in a non-constexpr context, will
  // trigger a runtime exception.
  explicit constexpr DebugInfoKind(value_type value) : value_(value) {
    if (value == nullptr) {
      // Since this function is not constexpr, it will trigger a compile-time
      // error if the DebugInfoKind() ctor is called in a constexpr context
      // with a null value.
      DebugInfoKind_ctor_parameter_must_not_be_null();
    }
  }

  // Predefined DebugInfoKinds for common use cases.
  // C++ doesn't allow declaring constexpr variables of an incomplete type,
  // so these are declared as const. However, they are defined as constexpr
  // outside the class, where the DebugInfoKind type is fully defined. We
  // do this to ensure these are available at static initialization time.
  static const DebugInfoKind PRODUCER_INFO;
  static const DebugInfoKind MOBILE_RUNTIME_INFO;
  static const DebugInfoKind PROFILER_STATE;
  static const DebugInfoKind INFERENCE_CONTEXT; // for inference usage
  static const DebugInfoKind PARAM_COMMS_INFO;
  static const DebugInfoKind TEST_INFO; // used only in tests
  static const DebugInfoKind TEST_INFO_2; // used only in tests

  // Comparison operators. These allow putting DebugInfoKind in containers like
  // std::set and std::map.
  constexpr bool operator==(const DebugInfoKind other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(const DebugInfoKind other) const {
    return value_ != other.value_;
  }
  constexpr bool operator<(const DebugInfoKind other) const {
    return value_ < other.value_;
  }

  C10_API friend std::ostream& operator<<(
      std::ostream& os,
      const DebugInfoKind& kind);

 private:
  // Doesn't compile in a constexpr context; throws in a non-constexpr context.
  // The function name is intentionally verbose to make the compiler error
  // more readable.
  [[noreturn]] static inline void
  DebugInfoKind_ctor_parameter_must_not_be_null() {
    TORCH_INTERNAL_ASSERT(
        false, "DebugInfoKind must be initialized with a non-null pointer.");
  }

  // Names for predefined DebugInfoKinds.
  inline static constexpr std::string_view kProducerInfoName = "PRODUCER_INFO";
  inline static constexpr std::string_view kMobileRuntimeInfoName =
      "MOBILE_RUNTIME_INFO";
  inline static constexpr std::string_view kProfilerStateName =
      "PROFILER_STATE";
  inline static constexpr std::string_view kInferenceContextName =
      "INFERENCE_CONTEXT";
  inline static constexpr std::string_view kParamCommsInfoName =
      "PARAM_COMMS_INFO";
  inline static constexpr std::string_view kTestInfoName = "TEST_INFO";
  inline static constexpr std::string_view kTestInfo2Name = "TEST_INFO_2";

  value_type value_ = nullptr;
};

inline constexpr DebugInfoKind DebugInfoKind::PRODUCER_INFO(&kProducerInfoName);
inline constexpr DebugInfoKind DebugInfoKind::MOBILE_RUNTIME_INFO(
    &kMobileRuntimeInfoName);
inline constexpr DebugInfoKind DebugInfoKind::PROFILER_STATE(
    &kProfilerStateName);
inline constexpr DebugInfoKind DebugInfoKind::INFERENCE_CONTEXT(
    &kInferenceContextName);
inline constexpr DebugInfoKind DebugInfoKind::PARAM_COMMS_INFO(
    &kParamCommsInfoName);
inline constexpr DebugInfoKind DebugInfoKind::TEST_INFO(&kTestInfoName);
inline constexpr DebugInfoKind DebugInfoKind::TEST_INFO_2(&kTestInfo2Name);

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
