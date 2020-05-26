#include <torch/csrc/jit/passes/utils/memory_dag.h>

#include <c10/util/flat_hash_map.h>
#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <queue>

namespace torch {
namespace jit {
namespace {
void makePointerToImpl(Element* from, Element* to) {
  from->pointsTo.set(to->index);
  to->pointedFrom.set(from->index);
}

Element* makeFreshValueImpl(
    const Value* v,
    std::vector<std::unique_ptr<Element>>& indexToElementMap_) {
  if (v == nullptr) {
    // Create a wildcard element, with no corresponding value
    indexToElementMap_.emplace_back(
        std::make_unique<Element>(indexToElementMap_.size()));
    return indexToElementMap_.back().get();
  }
  indexToElementMap_.emplace_back(
      std::make_unique<Element>(v, indexToElementMap_.size()));
  return indexToElementMap_.back().get();
}
} // namespace

Element::Element(const Value* value_, unsigned index_)
    : index(index_), values({value_}) {}
Element::Element(unsigned index_) : index(index_), values({}) {}

const Element* MemoryDAG::fromIndex(unsigned x) const {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

Element* MemoryDAG::fromIndex(unsigned x) {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = getMemoryLocations(a);
  const auto bMemLoc = getMemoryLocations(b);

  return aMemLoc.intersects(bMemLoc);
}

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return mayContainAliasImpl(a, b);
}

bool MemoryDAG::mayContainAlias(Element* a, Element* b) const {
  return mayContainAliasImpl(a, b);
}

void MemoryDAG::collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations& cont) const {
  // we have already recursed on this element
  unsigned compIdx = elem->index;
  if (cont.test(compIdx)) {
    return;
  }
  cont.set(compIdx);

  for (const auto& mem_loc : getMemoryLocations(elem)) {
    collectAllContainedMemoryLocations(fromIndex(mem_loc), cont);
  }

  for (const auto& contained : elem->containedElements) {
    collectAllContainedMemoryLocations(fromIndex(contained), cont);
  }
}

bool MemoryDAG::mayContainAliasImpl(const Element* a, const Element* b) const {
  MemoryLocations all_a_mlocs;
  MemoryLocations all_b_mlocs;

  collectAllContainedMemoryLocations(a, all_a_mlocs);
  collectAllContainedMemoryLocations(b, all_b_mlocs);

  return all_a_mlocs.intersects(all_b_mlocs);
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*> a,
    const at::ArrayRef<Element*> b) const {
  if (a.size() == 0 || b.size() == 0) {
    return false;
  }

  MemoryLocations all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  MemoryLocations all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  return all_a_mlocs.intersects(all_b_mlocs);
}

void MemoryDAGBuilder::makePointerTo(Element* from, Element* to) {
  makePointerToImpl(from, to);
}

void MemoryDAGBuilder::addToContainedElements(
    Element* elem,
    Element* container) {
  TORCH_INTERNAL_ASSERT(
      elem != container, "Elements cannot contain themselves");
  container->containedElements.set(elem->index);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAGBuilder::makeFreshValue(const Value* v) {
  return makeFreshValueImpl(v, indexToElementMap_);
}

const MemoryLocations& MemoryDAG::getMemoryLocations(const Element* e) const {
  // Note on cache invalidation: all mutation should occur through
  // MemoryDAGBuilder. Thus, once we consume the builder to create an immutable
  // MemoryDAG, we can cache here without worrying that we might potentially get
  // invalidated.
  if (e->cachedMemoryLocations_) {
    return *e->cachedMemoryLocations_;
  }

  MemoryLocations ret;
  if (e->pointsTo.empty()) {
    // Base case: if we don't point to anything, this element is a memory
    // location. Return itself.
    ret.set(e->index);
  } else {
    for (auto el : e->pointsTo) {
      ret |= getMemoryLocations(fromIndex(el));
    }
  }

  e->cachedMemoryLocations_ = std::move(ret);
  return *e->cachedMemoryLocations_;
}

void MemoryDAG::setWildcards(
    const std::unordered_set<const Value*>& wildcards,
    const ska::flat_hash_map<const Value*, Element*>& elementMap,
    const std::function<Element*(const Value*)>& getWildcardElement) {
  std::unordered_map<Element*, MemoryLocations> cacheUpdates;
  // If an element is set as a wildcard, that means that all its memory
  // locations must point to the wildcard element.
  for (const Value* v : wildcards) {
    auto wildcardElement = getWildcardElement(v);
    TORCH_INTERNAL_ASSERT(wildcardElement);

    const MemoryLocations pointeeSet = getMemoryLocations(elementMap.at(v));
    for (const auto& pointee : pointeeSet) {
      auto from = this->fromIndex(pointee);
      // avoid cycles where the wildcard points to itself
      if (from != wildcardElement) {
        makePointerToImpl(from, wildcardElement);
      }
    }

    // Track which memory locations we edited with a new pointer to the wildcard
    // element.
    cacheUpdates[wildcardElement] |= pointeeSet;
  }

  // Update caches in-place.
  // We take advantage of the fact that we only edited memory locations.
  //
  // Say we added a pointer from `MemoryLocationFoo -> WildcardBar`.
  // For every element, if the cache contains `MemoryLocationFoo`, then we must
  // add `WildcardBar` to it.
  for (const std::unique_ptr<Element>& e : this->indexToElementMap_) {
    if (e->values.empty()) {
      // This element is a wildcard element, we can skip it.
      TORCH_INTERNAL_ASSERT(e->pointsTo.empty());
      continue;
    }

    auto wildcardElement = getWildcardElement(*(e->values.begin()));
    if (!wildcardElement) {
      // This value is not a wildcard.
      continue;
    }
    auto it = cacheUpdates.find(wildcardElement);
    if (it == cacheUpdates.end()) {
      // We didn't rewrite any MemoryLocations to point to this element.
      continue;
    }
    // If this element contains an edited memory location, update the cache to
    // contain the pointed-to wildcard element as well.
    if (getMemoryLocations(e.get()).intersects(it->second)) {
      e->cachedMemoryLocations_->set(wildcardElement->index);
    }
  }
}

Element* MemoryDAG::unsafeMakeFreshValue(const Value* v) {
  return makeFreshValueImpl(v, indexToElementMap_);
}
} // namespace jit
} // namespace torch
