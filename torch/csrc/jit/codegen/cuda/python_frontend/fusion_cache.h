#pragma once
#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

#include <memory>

//! nvFuser Fusion IR namespace abbreviation
namespace Nvf = torch::jit::fuser::cuda;

namespace nvfuser {

struct RecordFunctor;

//! \struct FusionCacheEntry
//! \brief Is the container for a Node in the cache contained in the
//! FusionCache that is organized as a prefix tree.

struct TORCH_CUDA_CU_API FusionCacheEntry {
  FusionCacheEntry(RecordFunctor* rec, size_t _fusion_id = 0);

  // Queries whether the entry denotes a leaf node which also represents
  // a the end of Fusion entry in the cache.
  bool isTerminal() const;

  //! An entry's primary data is the record it holds
  std::unique_ptr<RecordFunctor> record;
  //! A hash map of the children for the current node.
  //! The hash map hashs a pointer to a RecordFunctor because
  //! the hash function is virtual.
  std::unordered_map<RecordFunctor*, std::unique_ptr<FusionCacheEntry>>
      record_hash_map;
  //! An index into FusionCache's vector of nvFuser object that holds an
  //! unscheduled Fusion.  The id is only valid if the entry is terminal.
  size_t fusion_id;
  //! Count of times the Entry is traversed
  size_t visits;
};

//! \class FusionCache
//! \brief A singleton class used in the nvFuser python interface
//! to manage the caching of fusions.
//!
//! The fusion cache implements a prefix tree of records in order to cache
//! fusions.  A leaf of the tree with a terminal node contains an nvFuser
//! Fusion IR container for a cached instance.
//!
//! \todo Add the ability to evict a fusion.  There is currently a max number
//! of fusions that is checked to prevent a runaway case.

class TORCH_CUDA_CU_API FusionCache {
  //! The constructor is private given the FusionCache is only constructed
  //! as a singleton.
  FusionCache(size_t max_fusions);

  //! Copy and Assignment of the FusionCache is not supported
  FusionCache(const FusionCache&) = delete;
  FusionCache& operator=(const FusionCache&) = delete;

 public:
  //! The next 2 pubic methods are the python interface methods

  //! Gets a pointer to the singleton and creates a new one if necessary
  static FusionCache* get(size_t max_fusions = 8192);
  //! Number of fusions cached
  size_t numFusions() const;
  //! print cache stats
  void print(std::ostream& os);
  //! Reset Cache to an empty state
  static void reset();

  //! The rest of the public methods are only used in C++

  //! Queries the current cache entry to see if a record matches one of its
  //! children
  c10::optional<FusionCacheEntry*> lookupFusionCacheEntry(
      RecordFunctor* rec) const;
  //! Creates a child node for the current cache entry and an optional
  //! fusion_id is returned if the new entry is terminal
  c10::optional<size_t> createFusionCacheEntry(RecordFunctor* rec);
  //! Resets the current cache pointer to the top of the tree
  void resetFusionCachePtr();
  //! Traverses the cache from the current entry to the child associated
  //! with the record given.
  void traverseFusionCache(RecordFunctor* rec);

  friend class FusionInterface;

 private:
  //! Returns the pointer to the current cache entry
  FusionCacheEntry* fusionCachePtr() const;

  //! The static pointer to the FusionCache
  static FusionCache* singleton_;

  //! The max allowed number of fusions in the cache
  size_t max_fusions_;
  //! The top of the prefix tree used to start a cache look up of a given
  //! fusion definition.
  std::unique_ptr<FusionCacheEntry> fusion_cache_start_;
  //! A pointer to the current cache entry in a cache lookup of a fusion
  //! definition.
  FusionCacheEntry* fusion_cache_ptr_;
  //! A vector of nvFuser Fusion IR fusions.
  std::vector<std::unique_ptr<Nvf::FusionExecutorCache>> fusions_;
  //! A vector of Terminal Cache Entries for Stats collection
  std::vector<FusionCacheEntry*> terminal_cache_entries_;
};

} // namespace nvfuser
