#include "memory_dag.h"

#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <queue>
#include <iostream>

namespace torch {
namespace jit {
ska::flat_hash_map<const Element*, int> comprMap;
ska::flat_hash_map<int, const Element*> decomprMap;

int getCompressed(const Element* x) {
  if (comprMap.count(x)) {
    return comprMap[x];
  }
  comprMap[x] = comprMap.size() + 1;
  decomprMap[comprMap.size()] = x;
  return comprMap[x];
}
const Element * getDecompressed(int x) {
  assert(decomprMap.count(x));
  return decomprMap[x];
}
bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::memoryLocationOverlap(
    const MemoryLocations & aMemLoc,
    const MemoryLocations & bMemLoc) const {
  return aMemLoc.intersects(bMemLoc);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = a->getMemoryLocations();
  const auto bMemLoc = b->getMemoryLocations();

  return memoryLocationOverlap(aMemLoc, bMemLoc);
}

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return mayContainAliasImpl(a, b);
}

bool MemoryDAG::mayContainAlias(Element* a, Element* b) const {
  return mayContainAliasImpl(a, b);
}

void collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations & cont) {
  // we have already recursed on this element
  int compIdx = getCompressed(elem);
  if (cont.test(compIdx)) {
    return;
  }

  cont.set(compIdx);

  for (const auto& mem_loc : elem->getMemoryLocations()) {
    collectAllContainedMemoryLocations(getDecompressed(mem_loc), cont);
  }

  for (const auto& contained : elem->contained_elements) {
    collectAllContainedMemoryLocations(getDecompressed(contained), cont);
  }
}

bool MemoryDAG::mayContainAliasImpl(const Element* a, const Element* b) const {
  MemoryLocations  all_a_mlocs;
  MemoryLocations  all_b_mlocs;

  collectAllContainedMemoryLocations(a, all_a_mlocs);
  collectAllContainedMemoryLocations(b, all_b_mlocs);

  return memoryLocationOverlap(all_a_mlocs, all_b_mlocs);
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*>& a,
    const at::ArrayRef<Element*>& b) const {
  if (a.size() == 0 || b.size() == 0) {
    return false;
  }

  MemoryLocations  all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  MemoryLocations  all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  return memoryLocationOverlap(all_a_mlocs, all_b_mlocs);
}

// Make `v` point at `to`.
void MemoryDAG::makePointerTo(Element* from, Element* to) {
  from->pointsTo.set(getCompressed(to));
  to->pointedFrom.set(getCompressed(from));
}

void MemoryDAG::addToContainedElements(Element* elem, Element* container) {
  container->contained_elements.set(getCompressed(elem));
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAG::makeFreshValue(const Value* v) {
  auto el = torch::make_unique<Element>();
  el->value = v;

  auto rawPtr = el.get();
  elements_.emplace(rawPtr, std::move(el));
  return rawPtr;
}

TORCH_API std::unordered_set<const Element*> convert(MemoryLocations bits) {
  std::unordered_set<const Element*> res;
  for (auto i: bits) {
    res.insert(getDecompressed(i));
  }
  return res;
}
const MemoryLocations& Element::getMemoryLocations() const {
  if (!cachedMemoryLocations_.empty()) {
    return cachedMemoryLocations_;
  }

  // Do a BFS in the `points-to` direction, collecting all memory locations
  MemoryLocations  ret;
  this->bfs(
      [&](const Element* el) {
        if (el->pointsTo.empty()) {
          ret.set(getCompressed(el));
        }
      },
      BfsDirection::POINTS_TO);
  cachedMemoryLocations_ = ret;
  return ret;
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
template <typename Fn>
bool Element::bfs(Fn fn, BfsDirection dir) const {
  std::queue<const Element*> queue;
  MemoryLocations  seen;
  queue.push(this);
  while (!queue.empty()) {
    const auto el = queue.front();
    queue.pop();
    seen.set(getCompressed(el));

    fn(el);

    switch (dir) {
      case BfsDirection::POINTS_TO: {
        for (auto ptr : el->pointsTo) {
          if (!seen.test(ptr)) {
            queue.push(getDecompressed(ptr));
          }
        }
      } break;

      case BfsDirection::POINTED_FROM: {
        for (auto ptr : el->pointedFrom) {
          if (!seen.test(ptr)) {
            queue.push(getDecompressed(ptr));
          }
        }
      } break;
    }
  }
  return false;
}
} // namespace jit
} // namespace torch
