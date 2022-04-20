#pragma once

#include <c10/util/Exception.h>

#include <algorithm>
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
std::string toString(Statement* stmt) {
  return stmt->toString();
}
} // namespace

//! Container class DisjointSet models equivalence relationships
//!
//! Each instance of this class keeps a set of equivalent classes
//! DisjointSet::join(a,b) makes the full class of a and b equivalent
//! DisjointSet::areEqual(a,b) checks if a and b belong same class
template <typename T, typename Hash = std::hash<T>>
class DisjointSet {
 public:
  DisjointSet() = default;

  //! Joins the equivalent class that a and b belong to
  //! areEqual(a',b') will be true for each a'=a and b'=b
  //!
  //! \param a An element from a equivalent class
  //!          will create a new equivalent class if a does
  //!          not belong to any
  //! \param b An element from another equivalent class
  //!          will create a new equivalent class if b does
  //!          not belong to any
  void join(T a, T b) {
    // cases where either of the quiv class doesn't exist
    if (!entry_map.count(a) && !entry_map.count(b)) {
      createPoint(a);
      entry_map[b] = fixedPoint(a);
    } else if (!entry_map.count(a)) {
      entry_map[a] = fixedPoint(b);
    } else if (!entry_map.count(b)) {
      entry_map[b] = fixedPoint(a);
    } else {
      // case where both equiv classes exist and need to join
      const int i0 = fixedPoint(a);
      const int i1 = fixedPoint(b);
      int new_parent = 0;
      int new_child = 0;

      // Either order here is correct but joining larger class to smaller class
      // tend to be faster
      std::tie(new_parent, new_child) = (weights[i0] < weights[i1])
          ? std::make_pair(i0, i1)
          : std::make_pair(i1, i0);
      weights[new_parent] += weights[new_child];
      set_map[new_child] = new_parent;
    }
  }

  //! Checks if a and b belong to the same equivalent class
  //!
  //! \param a An element from a equivalent class
  //! \param b An element from another equivalent class
  //! \returns Boolean value representing if a and b are
  //!          recorded to be in the same equivalent class
  //!          will return false if any of a or b doesn't
  //!          have an equivalent class recorded
  bool areEquivalent(T a, T b) const {
    if (!entry_map.count(a) || !entry_map.count(b)) {
      return false;
    }
    return fixedPoint(a) == fixedPoint(b);
  }

  //! Queries if an element exists in this set
  bool contains(T a) const {
    return entry_map.count(a) > 0;
  }

  //! Returns all elements added to this set
  std::vector<T> getAllElements() const {
    std::vector<T> elms(entry_map.size());
    std::transform(
        entry_map.begin(),
        entry_map.end(),
        elms.begin(),
        [](const auto& entry_map_kv) { return entry_map_kv.first; });
    return elms;
  }

  //! Clears the equivalence relationships
  void clear() {
    set_map.clear();
    weights.clear();
    entry_map.clear();
    next_index_ = 0;
  }

  //! Generates all the disjoint sets as maps and returns them.
  // TODO: unify fixed_point_map creation with print.
  std::vector<std::unordered_set<T, Hash>> generateDisjointSets() const {
    std::unordered_map<int, std::unordered_set<T, Hash>> fixed_point_map;
    int num_sets = 0;
    for (const auto& kv : entry_map) {
      int fixed_point = fixedPoint(kv.first);
      num_sets = std::max(num_sets, fixed_point + 1);
      auto it = fixed_point_map.find(fixed_point);
      if (it == fixed_point_map.end()) {
        it = fixed_point_map.insert({fixed_point, {}}).first;
      }
      it->second.insert(kv.first);
    }

    std::vector<std::unordered_set<T, Hash>> disjoint_sets;
    disjoint_sets.resize(num_sets);
    for (auto entry : fixed_point_map) {
      TORCH_INTERNAL_ASSERT(
          entry.first < num_sets, "Unexpected error generating disjoint sets.");
      disjoint_sets[entry.first] = entry.second;
    }

    // Vector may contain some empty sets, remove them
    disjoint_sets.erase(
        std::remove_if(
            disjoint_sets.begin(),
            disjoint_sets.end(),
            [](std::unordered_set<T, Hash> set) { return set.empty(); }),
        disjoint_sets.end());

    return disjoint_sets;
  }

  //! Dumps the equivalent relationships
  // TOOD: Change to "toString" for consistency
  std::ostream& print(std::ostream& os) const {
    std::unordered_map<int, std::unordered_set<T, Hash>> fixed_point_map;
    for (const auto& kv : entry_map) {
      int fixed_point = fixedPoint(kv.first);
      auto it = fixed_point_map.find(fixed_point);
      if (it == fixed_point_map.end()) {
        it = fixed_point_map.insert({fixed_point, {}}).first;
      }
      it->second.insert(kv.first);
    }
    os << "{\n";
    for (const auto& kv : fixed_point_map) {
      os << "\t{ ";
      for (const auto& val : kv.second) {
        // TODO: Fix printing to avoid trailing ; Tried using std::prev to
        // easily identify the last val, but that seems to have modified the
        // .end() value in kv.second and ended in an infinite loop
        os << toString(val) << "; ";
      }
      os << "}\n";
    }
    os << "}\n";
    return os;
  }

 private:
  // Internal fixed point implementation:
  //  Returns the equivalent class that e belongs to
  int getFixedPointForClass(int e) const {
    TORCH_INTERNAL_ASSERT(static_cast<int>(set_map.size()) > e);
    while (set_map[e] != e) {
      // Chasing to fixed point
      e = set_map[e];
    }
    return e;
  }

  //! Utility to check the class e belongs to:
  //!
  //! \param e element e to find the equiv class for
  //! \returns the equivalent class that e belongs to
  //!
  int fixedPoint(T e) const {
    // Handles case when i doesn't have an equivalence class
    TORCH_INTERNAL_ASSERT(entry_map.count(e));

    // Use fixed point as a representation for the equiv class
    return getFixedPointForClass(entry_map.at(e));
  }

  //! Utility to create a new equiv class for i
  //
  //! \param i Element i to create the equiv class for
  void createPoint(T i) {
    entry_map[i] = next_index_;
    set_map.push_back(next_index_++);
    weights.push_back(1);
  }

 private:
  // Internal representation of the equivalence class as integers
  // set_map implements the "parent" relationship
  std::vector<int> set_map;
  // Weights is used for preliminary perf optimization
  std::vector<int> weights;

  // Map the input of type T to its equivalence class
  std::unordered_map<T, int, Hash> entry_map;

  // Running counter for generating new index when
  // Creating new equiv classes
  int next_index_ = 0;
};

// Vector like class that will prevent adding duplicate entries by also
// maintaing a set
template <typename T>
class UniquePtrVector {
 public:
  // Returns if a node was actually added
  bool pushBack(T* entry) {
    if (set_.emplace(entry).second) {
      vector_.push_back(entry);
      return true;
    }
    return false;
  }

  // Returns if any node was added
  bool pushBack(const UniquePtrVector<T>& other) {
    bool any_added = false;
    for (auto entry : other) {
      any_added = any_added | pushBack(entry);
    }
    return any_added;
  }

  const std::vector<T*>& vector() const {
    return vector_;
  }

  T* back() const {
    return vector_.back();
  }

  bool has(T* entry) const {
    return set_.find(entry) != set_.end();
  }

  std::string toString() {
    std::stringstream ss;
    ss << "{ ";
    for (auto entry : vector()) {
      ss << entry->toString();
      if (entry != vector().back()) {
        ss << "; ";
      }
    }
    ss << " }";
    return ss.str();
  }

 private:
  std::vector<T*> vector_;
  std::unordered_set<T*> set_;
};

// Disjoint set using the approach originally from compute at map
//
// TODO: Unify with the above disjoint set in the codebase.
template <typename T, typename Hash = std::hash<T>>
class DisjointSetsOfPointers {
 public:
  DisjointSetsOfPointers() = default;

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as UniquePtrVector is not returned as a const.
  const std::unordered_map<T*, std::shared_ptr<UniquePtrVector<T>>>&
  disjointSetMap() const {
    return disjoint_set_maps_;
  }

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as UniquePtrVector is not returned as a const.
  const std::vector<std::shared_ptr<UniquePtrVector<T>>>& disjointSets() const {
    return disjoint_sets_;
  }

  const UniquePtrVector<T*>& getDisjointSetOf(T* entry) const {
    auto set_it = disjoint_set_maps_.find(entry);
    TORCH_INTERNAL_ASSERT(
        set_it != disjoint_set_maps_.end(),
        "Could not find entry for ",
        entry->toString());
    return *set_it.second;
  }

  // TODO: Return iterator
  void initializeSet(T* entry) {
    disjoint_sets_.push_back(std::make_shared<UniquePtrVector<T>>());
    disjoint_sets_.back()->pushBack(entry);
    disjoint_set_maps_.emplace(std::make_pair(entry, disjoint_sets_.back()));
  }

  void mapEntries(T* entry0, T* entry1) {
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

  std::string toString() const {
    std::stringstream ss;
    ss << "disjoint sets{\n";
    for (auto s_ptr : disjoint_sets_) {
      auto& set = *s_ptr;
      ss << "  { ";
      for (auto entry : set.vector()) {
        ss << entry->toString();
        if (entry != set.back()) {
          ss << "; ";
        }
      }
      ss << " }\n";
    }
    ss << "}";
    return ss.str();
  }

 private:
  // Disjoint sets
  std::unordered_map<T*, std::shared_ptr<UniquePtrVector<T>>>
      disjoint_set_maps_;

  // Keep a list of disjoint_sets that's deterministic to iterate over
  std::vector<std::shared_ptr<UniquePtrVector<T>>> disjoint_sets_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
