#pragma once

#include <cstdint>

namespace c10 {
namespace impl {
class OperatorEntry;
}

enum class AliasAnalysisKind : uint8_t {
  INTERNAL_SPECIAL_CASE,
  CONSERVATIVE, // The most conservative alias analysis type, assumes
                // side-effects. This is the default analysis.
  FROM_SCHEMA,
  PURE_FUNCTION
};

#if !defined(_MSC_VER)
constexpr // Our current MSVC version has a bug that doesn't allow this to be constexpr.
#endif
inline const char* toString(AliasAnalysisKind aliasAnalysisKind) {
  return (aliasAnalysisKind == AliasAnalysisKind::CONSERVATIVE)
      ? "CONSERVATIVE"
      : (aliasAnalysisKind == AliasAnalysisKind::FROM_SCHEMA)
          ? "FROM_SCHEMA"
          : (aliasAnalysisKind == AliasAnalysisKind::PURE_FUNCTION)
              ? "PURE_FUNCTION"
              : (aliasAnalysisKind == AliasAnalysisKind::INTERNAL_SPECIAL_CASE)
                  ? "INTERNAL_SPECIAL_CASE"
                  : "UNKNOWN";
}

struct OperatorOptions final {
public:
  bool isDefaultAliasAnalysisKind() const {
    return aliasAnalysisKind_ == c10::nullopt;
  }

  AliasAnalysisKind aliasAnalysis() const {
    return !isDefaultAliasAnalysisKind()
      ? *aliasAnalysisKind_
      : AliasAnalysisKind::CONSERVATIVE;
  }

  void setAliasAnalysis(AliasAnalysisKind v) {
    aliasAnalysisKind_ = v;
  }

  friend bool operator==(const OperatorOptions& lhs, const OperatorOptions& rhs) {
    return lhs.aliasAnalysisKind_ == rhs.aliasAnalysisKind_;
  }

  friend bool operator!=(const OperatorOptions& lhs, const OperatorOptions& rhs) {
    return !(lhs == rhs);
  }

private:
 c10::optional<AliasAnalysisKind> aliasAnalysisKind_;
};

} // namespace c10
