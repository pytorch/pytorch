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
//! FusionManager that is organized as a prefix tree.

struct TORCH_CUDA_CU_API FusionCacheEntry {
  FusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  FusionCacheEntry();

  //! An entry's primary data is the record it holds
  std::shared_ptr<RecordFunctor> record;
  //! A hash map of the children for the current node.
  //! The hash map hashs a pointer to a RecordFunctor because
  //! the hash function is virtual.  Also, a shared_ptr is used
  //! to own the pointer because the original owner of a pointer,
  //! a FusionDefintion, can't be relied upon to hold
  //! hold the record over the lifetime of the FusionManager.
  std::unordered_map<
      std::shared_ptr<RecordFunctor>,
      std::unique_ptr<FusionCacheEntry>>
      record_hash_map;

  //! This boolean indicates a leaf node with a cached nvFuser Fusion
  bool is_terminal;
  //! A pointer to the nvFuser object that holds an unscheduled Fusion
  std::unique_ptr<Nvf::FusionExecutorCache> fusion_executor_cache;
};

//! \class FusionManager
//! \brief A singleton class used in the nvFuser python interface
//! to manage the caching of fusions.
//!
//! Example:
//!
//!   fm = FusionManager.get()
//!
//!   with FusionDefinition(fm) as fd :
//!       t0 = fd.define_tensor(3)
//!       s1 = fd.define_constant(3.)
//!       t2 = fd.ops.add(t0, s1)
//!       fd.add_output(t2)
//!
//!   input = torch.ones(2, 4, 8, device='cuda')
//!
//!   for _ in range(5) :
//!      outputs = fm.execute([input])
//!
//! The python class defintion is effecively:
//! class FusionManager :
//!     # Static Methods
//!     def get(max_fusions=4096):
//!        ....
//!     def reset():
//!        ....
//!     # Methods
//!     def execute(inputs):
//!        ....
//!     def print_ir():
//!        ....
//!
//! The fusion manager implements a prefix tree of records in order to cache
//! fusions.  A leaf of the tree with a terminal node contains an nvFuser
//! Fusion IR container for a cached instance.
//!
//! \todo Add the ability to evict a fusion.  There is currently a max number
//! of fusions that is checked to prevent a runaway case.

class TORCH_CUDA_CU_API FusionManager {
  //! The constructor is private given the FusionManager is only constructed
  //! as a singleton.
  FusionManager(size_t max_fusions);

  //! Copy and Assignment of the FusionManager is not supported
  FusionManager(const FusionManager&) = delete;
  FusionManager& operator=(const FusionManager&) = delete;

 public:
  //! The next 4 pubic methods are the python interface methods

  //! Gets a pointer to the singleton and creates a new one if necessary
  static FusionManager* get(size_t max_fusions);
  //! Executes a fusion if the current cache pointer points at a terminal node
  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs);
  //! Prints the nvFuser IR if the current cache pointer is a terminal node
  void printIr() const;

  //! The rest of the public methods are only used in C++

  //! Returns a pointer for the Fusion associated with the current cache
  //! pointer if the current cache entry is a terminal node.
  Nvf::Fusion* fusionPtr() const;
  //! Queries the current cache entry to see if a record matches one of its
  //! children
  c10::optional<FusionCacheEntry*> lookupFusionCacheEntry(
      std::shared_ptr<RecordFunctor>& rec) const;
  //! Creates a child node for the current cache entry
  void createFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  //! Creates a child node for the current cache entry that is terminal
  void createTerminalFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  //! Resets the current cache pointer to the top of the tree
  void resetFusionCachePtr();
  //! Traverses the cache from the current entry to the child associated
  //! with the record given.
  void traverseFusionCache(std::shared_ptr<RecordFunctor>& rec);

 private:
  //! Gives a pointer to the FusionExecutorCache associated with the
  //! current cache entry if it is a terminal node.
  Nvf::FusionExecutorCache* fusionExecutorCachePtr() const;
  //! Returns the pointer to the current cache entry
  FusionCacheEntry* fusionCachePtr() const;

  //! The static pointer to the FusionManager
  static thread_local FusionManager* singleton_;

  //! The max allowed number of fusions in the cache
  size_t max_fusions_;
  //! The current number of fusions in the cache.
  size_t num_fusions_;
  //! A dummy record for start of the fusion cache tree.
  std::shared_ptr<RecordFunctor> start_record_;
  //! The top of the prefix tree used to start a cache look up of a given
  //! fusion definition.
  std::unique_ptr<FusionCacheEntry> fusion_cache_start_;
  //! A pointer to the current cache entry in a cache lookup of a fusion
  //! definition.
  FusionCacheEntry* fusion_cache_ptr_;
};

} // namespace nvfuser
