#pragma once
#include <unordered_set>
#include <vector>
#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>

namespace c10 {
class AliasInfo {
 public:
  // Symbol for the set that can alias anything
  static Symbol wildcardSet() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }
  static AliasInfo createWildcard() {
    AliasInfo ret;
    ret.addSet(wildcardSet());
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
    AT_ASSERT(sets_.size() == 1);
    return *sets_.begin();
  }

  bool isWildcard() const {
    return sets_.count(wildcardSet()) != 0;
  }

  void unionWith(const AliasInfo& other) {
    for (const auto& alias : other.sets()) {
      sets_.insert(alias);
    }
  }

  // TODO this doesn't check any contained types yet
  // non-strict: returns true if self.sets() == other.sets()
  bool isSubsetOf(const AliasInfo& other) const {
    for (const auto& alias : this->sets()) {
      if (other.sets().count(alias) == 0) {
        return false;
      }
    }
    return true;
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
