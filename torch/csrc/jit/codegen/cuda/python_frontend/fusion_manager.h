#pragma once
#include <memory>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

//! nvFuser Fusion IR Types
using NvfFusionExecutorCache = torch::jit::fuser::cuda::FusionExecutorCache;
using NvfFusion = torch::jit::fuser::cuda::Fusion;

namespace nvfuser {

struct RecordFunctor;

struct FusionCacheEntry {
  FusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  FusionCacheEntry();

  std::shared_ptr<RecordFunctor> record;
  std::unordered_map<RecordFunctor*, std::unique_ptr<FusionCacheEntry>>
      record_hash_map;

  bool is_terminal;
  std::unique_ptr<NvfFusionExecutorCache> fusion_executor_cache;
};

class FusionManager {
 public:
  FusionManager();

  //! Copy and Assignment of the FusionManager is not supported
  FusionManager(const FusionManager&) = delete;
  FusionManager& operator=(const FusionManager&) = delete;

  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs);
  void printIr() const;
  void printKernel() const;
  NvfFusion* fusionPtr() const;

  c10::optional<FusionCacheEntry*> lookupFusionCacheEntry(
      RecordFunctor* rec) const;
  void createFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  void createTerminalFusionCacheEntry(std::shared_ptr<RecordFunctor>& rec);
  void resetFusionCachePtr();
  void traverseFusionCache(std::shared_ptr<RecordFunctor>& rec);

 private:
  NvfFusionExecutorCache* fusionExecutorCachePtr() const;

  //! The fusion cache is implemented as a prefix tree of entries containing
  //! a Record representing a Fusion Definition line entry.
  std::shared_ptr<RecordFunctor> start_record_;
  std::unique_ptr<FusionCacheEntry> fusion_cache_start_;
  FusionCacheEntry* fusion_cache_ptr_;
};

} // namespace nvfuser
