#pragma once
#include <memory>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

//! nvFuser Fusion IR namespace abbreviation
namespace Nvf = torch::jit::fuser::cuda;

namespace nvfuser {

struct RecordFunctor;

struct FusionCacheEntry {
  FusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  FusionCacheEntry();

  std::shared_ptr<RecordFunctor> record;
  std::unordered_map<
      std::shared_ptr<RecordFunctor>,
      std::unique_ptr<FusionCacheEntry>>
      record_hash_map;

  bool is_terminal;
  std::unique_ptr<Nvf::FusionExecutorCache> fusion_executor_cache;
};

class FusionManager {
  FusionManager(size_t max_fusions);
  
  //! Copy and Assignment of the FusionManager is not supported
  FusionManager(const FusionManager&) = delete;
  FusionManager& operator=(const FusionManager&) = delete;

 public:
  static FusionManager* get(size_t max_fusions);
  static void reset();

  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs);
  void printIr() const;
  Nvf::Fusion* fusionPtr() const;

  c10::optional<FusionCacheEntry*> lookupFusionCacheEntry(
      std::shared_ptr<RecordFunctor>& rec) const;
  void createFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  void createTerminalFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  void resetFusionCachePtr();
  void traverseFusionCache(std::shared_ptr<RecordFunctor>& rec);

 private:
  Nvf::FusionExecutorCache* fusionExecutorCachePtr() const;
  FusionCacheEntry* fusionCachePtr() const;

  static thread_local FusionManager* singleton_;

  size_t max_fusions_;
  size_t num_fusions_;
  //! The fusion cache is implemented as a prefix tree of entries containing
  //! a Record representing a Fusion Definition line entry.
  std::shared_ptr<RecordFunctor> start_record_;
  std::unique_ptr<FusionCacheEntry> fusion_cache_start_;
  FusionCacheEntry* fusion_cache_ptr_;
};

} // namespace nvfuser
