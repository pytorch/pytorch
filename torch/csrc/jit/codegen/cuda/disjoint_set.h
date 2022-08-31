#pragma once

#include <c10/util/Exception.h>

#include <algorithm>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// For printing of the set when using a Statement as the type for the set
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

template <typename T>
std::string abstractToString(T* ptr) {
  return ptr->toString();
}

template <typename T>
std::string abstractToString(T ref) {
  return ref.toString();
}

} // namespace

// Vector like class that will prevent adding duplicate entries by also
// maintaing a set
template <typename T, typename Hash = std::hash<T>>
class VectorOfUniqueEntries {
 public:
  VectorOfUniqueEntries() = default;

  VectorOfUniqueEntries(const std::initializer_list<T>& x)
      : vector_(x), set_(x) {}

  // Returns if a node was actually added
  bool pushBack(T entry) {
    if (set_.emplace(entry).second) {
      vector_.push_back(entry);
      return true;
    }
    return false;
  }

  // Returns if any node was added
  bool pushBack(const VectorOfUniqueEntries<T, Hash>& other) {
    bool any_added = false;
    for (auto entry : other) {
      any_added = any_added | pushBack(entry);
    }
    return any_added;
  }

  // Returns a const vector useful for iterating on
  const std::vector<T>& vector() const {
    return vector_;
  }

  // Returns first element in vector
  T front() const {
    return vector_.front();
  }

  // Returns last element in vector
  T back() const {
    return vector_.back();
  }

  // Remove and returns the last element in vector
  T popBack() {
    T v = vector_.back();
    set_.erase(v);
    vector_.pop_back();
    return v;
  }

  // Returns if this container is empty
  bool empty() const {
    return vector_.empty();
  }

  // Returns the number of elements in this container
  size_t size() const {
    return vector_.size();
  }

  // Returns if entry is in this vector
  bool has(T entry) const {
    return set_.find(entry) != set_.end();
  }

  // Erase given entry from the containers if
  //  there is a match.
  void erase(T entry) {
    vector_.erase(
        std::remove_if(
            vector_.begin(),
            vector_.end(),
            [entry](T val) { return val == entry; }),
        vector_.end());

    set_.erase(entry);
  }

  // Insert elements at the end of the container.
  template <typename InputIt>
  void insert(InputIt begin, InputIt end) {
    for (auto it = begin; it != end; it++) {
      pushBack(*it);
    }
  }

  // Returns iterator pointing to the beginning of vector container
  auto begin() const {
    return vector().begin();
  }

  // Returns iterator pointing to the end of vector container
  auto end() const {
    return vector().end();
  }

  // Returns iterator pointing to the beginning of vector container
  auto begin() {
    return vector().begin();
  }

  // Returns iterator pointing to the end of vector container
  auto end() {
    return vector().end();
  }

  std::string toString() {
    std::stringstream ss;
    ss << "{ ";
    for (auto entry : vector()) {
      ss << abstractToString(entry);
      if (entry != vector().back()) {
        ss << "; ";
      }
    }
    ss << " }";
    return ss.str();
  }

 private:
  std::vector<T> vector_;
  std::unordered_set<T, Hash> set_;
};

//! Container class DisjointSet models equivalence relationships
//!
//! Each instance of this class keeps equivalence sets
//! DisjointSet::mapEntries(a,b) makes the full set of a and b equivalent
//! DisjointSet::*AreMapped(a,b) checks if a and b belong to the same disjoint
//! set
template <typename T, typename Hash = std::hash<T>>
class DisjointSets {
 public:
  DisjointSets() = default;

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as VectorOfUniqueEntries is not returned as a const.
  const std::
      unordered_map<T, std::shared_ptr<VectorOfUniqueEntries<T, Hash>>, Hash>&
      disjointSetMap() const {
    return disjoint_set_maps_;
  }

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as VectorOfUniqueEntries is not returned as a const.
  const std::vector<std::shared_ptr<VectorOfUniqueEntries<T, Hash>>>&
  disjointSets() const {
    return disjoint_sets_;
  }

  // Return the entire disjoint set of provided entry
  const VectorOfUniqueEntries<T, Hash>& getDisjointSetOf(T entry) const {
    auto set_it = disjoint_set_maps_.find(entry);
    TORCH_INTERNAL_ASSERT(
        set_it != disjoint_set_maps_.end(),
        "Could not find entry for ",
        entry->toString());
    return *(set_it->second);
  }

  // Initializes a new set for provided entry
  //
  // TODO: Return iterator
  void initializeSet(T entry) {
    if (disjoint_set_maps_.find(entry) != disjoint_set_maps_.end()) {
      return;
    }

    disjoint_sets_.push_back(
        std::make_shared<VectorOfUniqueEntries<T, Hash>>());
    disjoint_sets_.back()->pushBack(entry);
    disjoint_set_maps_.emplace(std::make_pair(entry, disjoint_sets_.back()));
  }

  // Adds all of the disjoint set belonging to entry1 to the disjoint set
  // belonging to entry0, maps all entries of disjoint set belonging to entry1
  // to entry0, removes original disjoint set belonging to entry1.
  void mapEntries(T entry0, T entry1) {
    auto set_it_0 = disjoint_set_maps_.find(entry0);
    auto set_it_1 = disjoint_set_maps_.find(entry1);

    // Track if we need to reset iterators, optimize for case where both entries
    // exist
    bool invalid_iterators = false;
    if (set_it_0 == disjoint_set_maps_.end()) {
      initializeSet(entry0);
      invalid_iterators = true;
    }

    if (set_it_1 == disjoint_set_maps_.end()) {
      initializeSet(entry1);
      invalid_iterators = true;
    }

    // TODO: We can avoid refinding one iterator if initialize set returns an
    // iterator, though if we insert entry1 we'd have to refind entry0 as it
    // could invalidate all iterators
    if (invalid_iterators) {
      set_it_0 = disjoint_set_maps_.find(entry0);
      set_it_1 = disjoint_set_maps_.find(entry1);
    }

    auto set0_shared_ptr = set_it_0->second;
    auto set1_shared_ptr = set_it_1->second;

    // If the sets are already the same, do nothing
    if (set0_shared_ptr == set1_shared_ptr) {
      return;
    }

    // Place everything in set1 into set0 and remap all entries in set1 to set0
    for (auto entry : set1_shared_ptr->vector()) {
      set0_shared_ptr->pushBack(entry);
      disjoint_set_maps_[entry] = set0_shared_ptr;
    }

    // set1 no longer needed as its entries are copied into set0
    disjoint_sets_.erase(std::find(
        disjoint_sets_.begin(), disjoint_sets_.end(), set1_shared_ptr));
  }

  // Will assert if provided entry0 is not in any disjoint set, otherwise
  // returns if entry0 and entry1 are in the same disjoint set.
  bool strictAreMapped(T entry0, T entry1) const {
    auto entry_it = disjointSetMap().find(entry0);
    TORCH_INTERNAL_ASSERT(
        entry_it != disjointSetMap().end(),
        "Strict mapping failed on element: ",
        abstractToString(entry0),
        " either an error occured, or non strict mapping should have been used.");
    return entry_it->second->has(entry1);
  }

  // If entry0 doesn't have a disjoint set returns false, otherwise returns if
  // entry0 and entry1 are in the same disjoint set.
  bool permissiveAreMapped(T entry0, T entry1) const {
    auto entry_it = disjointSetMap().find(entry0);
    if (entry_it == disjointSetMap().end()) {
      return false;
    }
    return entry_it->second->has(entry1);
  }

  // Returns if a set exists with provided entry
  bool mappingExists(T entry) const {
    return disjoint_set_maps_.find(entry) != disjoint_set_maps_.end();
  }

  // Returns a deterministic list of all entries that have been added to any
  // disjoint set.
  //
  // Warning: constructed on every call, consider caching result.
  VectorOfUniqueEntries<T, Hash> getAllElements() const {
    VectorOfUniqueEntries<T, Hash> all_elements;
    for (auto set : disjoint_sets_) {
      for (auto entry : set->vector()) {
        all_elements.pushBack(entry);
      }
    }
    return all_elements;
  }

  // Completely clears all disjoint sets
  void clear() {
    disjoint_set_maps_.clear();
    disjoint_sets_.clear();
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "disjoint sets{\n";
    const std::string sep("  ");
    for (auto s_ptr : disjoint_sets_) {
      auto& set = *s_ptr;
      ss << sep << "{\n";
      for (auto entry : set.vector()) {
        ss << sep << sep << abstractToString(entry) << "\n";
      }
      ss << sep << "}\n";
    }
    ss << "}";
    return ss.str();
  }

 private:
  // Disjoint sets
  std::unordered_map<T, std::shared_ptr<VectorOfUniqueEntries<T, Hash>>, Hash>
      disjoint_set_maps_;

  // Keep a list of disjoint_sets that's deterministic to iterate over
  std::vector<std::shared_ptr<VectorOfUniqueEntries<T, Hash>>> disjoint_sets_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
