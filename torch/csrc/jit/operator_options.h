#pragma once

#include <torch/csrc/jit/passes/alias_analysis.h>

namespace torch {
namespace jit {

enum class AliasAnalysisKind {
  DEFAULT, // The most conservative alias analysis type, assumes side-effects
  PURE
};

struct OperatorOptions {
  OperatorOptions(){};

  OperatorOptions aliasAnalysis(AliasAnalysisKind aak) const noexcept {
    OperatorOptions r = *this;
    r.aliasAnalysisKind_ = aak;
    return r;
  }

  const AliasAnalysisKind& aliasAnalysis() const {
    return aliasAnalysisKind_;
  }
  AliasAnalysisKind aliasAnalysisKind_ = AliasAnalysisKind::DEFAULT;
};

} // namespace jit
} // namespace torch
