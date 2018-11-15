#pragma once
#include <vector>
#include <ATen/core/interned_strings.h>

namespace c10 {

struct AliasInfo {
  // Symbol for the set that can alias
  static Symbol wildcard() {
     static const Symbol wc = Symbol::fromQualString("alias::*");
     return wc;
  }
  AliasInfo(std::vector<Symbol> sets = {},
            std::vector<AliasInfo> contained_types = {})
  : sets_(std::move(sets))
  , contained_types_(std::move(contained_types)) {}


  // all sets it can be in
  const std::vector<Symbol>& sets() {
    return sets_;
  }
  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  const std::vector<AliasInfo>& containedTypes() {
    return contained_types_;
  }

private:
  std::vector<Symbol> sets_;
  std::vector<AliasInfo> contained_types_;
};

} // namespace c10
