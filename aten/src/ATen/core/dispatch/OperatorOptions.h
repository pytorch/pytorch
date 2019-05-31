#pragma once

#include <cstdint>

namespace c10 {
namespace impl {
class OperatorEntry;
}

enum class AliasAnalysisKind : uint8_t {
  DEFAULT, // The most conservative alias analysis type, assumes side-effects
  PURE
};

struct OperatorOptions final {
public:
  AliasAnalysisKind aliasAnalysis() const {
    return aliasAnalysisKind_;
  }

  void setAliasAnalysis(AliasAnalysisKind v) {
    aliasAnalysisKind_ = v;
  }

private:
  AliasAnalysisKind aliasAnalysisKind_ = AliasAnalysisKind::DEFAULT;
};

} // namespace c10
