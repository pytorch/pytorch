#pragma once
#include <unordered_set>
#include <vector>
#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>

namespace c10 {
/**
 * class AliasInfo
 *
 * Data structure to hold aliasing information for an `Argument`. They can be
 * nested to represent aliasing information on contained types.
 *
 * There is a `beforeSet` which describes the aliasing information before the
 * operator executes, and an `afterSet` that describes aliasing info
 * after execution.
 */
class AliasInfo {
 public:
  // Symbol for the set that can alias anything
  static Symbol wildcardSet() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }

  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }

  bool isWrite() const {
    return isWrite_;
  }

  void addBeforeSet(Symbol aliasSet) {
    beforeSets_.insert(aliasSet);
  }

  void addAfterSet(Symbol aliasSet) {
    afterSets_.insert(aliasSet);
  }

  const std::unordered_set<Symbol>& beforeSets() const {
    return beforeSets_;
  }

  const std::unordered_set<Symbol>& afterSets() const {
    return afterSets_;
  }

  Symbol beforeSet() const {
    AT_ASSERT(beforeSets_.size() == 1);
    return *beforeSets_.begin();
  }

  bool isWildcardBefore() const {
    return beforeSets_.count(wildcardSet()) != 0;
  }

  bool isWildcardAfter() const {
    return afterSets_.count(wildcardSet()) != 0;
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
  std::unordered_set<Symbol> beforeSets_;
  std::unordered_set<Symbol> afterSets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

inline bool operator==(const AliasInfo& lhs, const AliasInfo& rhs) {
  return lhs.isWrite() == rhs.isWrite()
      && lhs.beforeSets() == rhs.beforeSets()
      && lhs.afterSets() == rhs.afterSets()
      && lhs.containedTypes() == rhs.containedTypes();
}

// DEBUG ONLY; this does not match the way things are represented in the schema
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {
  out << "(";
  bool first = true;
  for (const auto& set : aliasInfo.beforeSets()) {
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
