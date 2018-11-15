#pragma once
#include <unordered_set>
#include <vector>
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/interned_strings.h"

namespace c10 {
class AliasInfo {
 public:
  // Symbol for the set that can alias
  static AliasInfo createWildcard() {
    AliasInfo ret;
    ret.addSet(wildcard());
    return ret;
  }

  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }

  bool isWrite() const {
    return isWrite_;
  }

  void addSet(Symbol aliasSet) {
    sets_.insert(aliasSet);
  }

  const std::unordered_set<Symbol>& sets() const {
    return sets_;
  }

  Symbol set() const {
    JIT_ASSERT(sets_.size() == 1);
    return *sets_.begin();
  }

  bool isWildcard() const {
    return sets_.count(wildcard()) != 0;
  }

  void unionWith(const AliasInfo& other) {
    for (const auto& alias : other.sets()) {
      sets_.insert(alias);
    }
  }
  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  static Symbol wildcard() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }
  std::unordered_set<Symbol> sets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

// DEBUG ONLY; this does not match the way things are represented in the schema
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {
  out << "(";
  bool first = true;
  for (const auto& set : aliasInfo.sets()) {
    if (first) {
      first = false;
    } else {
      out << "|";
    }
    out << set.toUnqualString();
  }
  out << ")";

  if (!aliasInfo.containedTypes().empty()) {
    out << " CONTAINS " << aliasInfo.containedTypes()[0];
  }
  return out;
}
} // namespace c10
