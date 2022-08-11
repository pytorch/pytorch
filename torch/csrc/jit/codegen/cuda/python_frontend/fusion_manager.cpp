#ifdef USE_CUDA
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

namespace nvfuser {

FusionCacheEntry::FusionCacheEntry(std::shared_ptr<RecordFunctor>& rec)
    : record(rec),
      record_hash_map(),
      is_terminal(false),
      fusion_executor_cache(nullptr) {}
FusionCacheEntry::FusionCacheEntry()
    : record(new EndRecord()),
      record_hash_map(),
      is_terminal(true),
      fusion_executor_cache(std::make_unique<Nvf::FusionExecutorCache>(
          std::make_unique<Nvf::Fusion>())) {}

FusionManager::FusionManager(size_t max_fusions)
    : max_fusions_(max_fusions),
      num_fusions_(0),
      start_record_(new StartRecord()),
      fusion_cache_start_(new FusionCacheEntry(start_record_)),
      fusion_cache_ptr_(fusion_cache_start_.get()) {}

std::vector<at::Tensor> FusionManager::execute(
    const at::ArrayRef<c10::IValue>& inputs) {
  return fusionExecutorCachePtr()->runFusionWithInputs(inputs);
}
void FusionManager::printIr() const {
  fusionExecutorCachePtr()->printFusion();
}
void FusionManager::printKernel() const {
  fusionPtr()->printKernel();
}

c10::optional<FusionCacheEntry*> FusionManager::lookupFusionCacheEntry(
    std::shared_ptr<RecordFunctor>& rec) const {
  auto cache_entry = fusionCachePtr()->record_hash_map.find(rec);
  if (cache_entry == std::end(fusion_cache_ptr_->record_hash_map)) {
    return c10::nullopt;
  } else {
    if (cache_entry->second) {
      return c10::optional<FusionCacheEntry*>(cache_entry->second.get());
    } else {
      TORCH_INTERNAL_ASSERT(false, "The looked up cache entry is null.");
      return c10::nullopt;
    }
  }
}
void FusionManager::createFusionCacheEntry(
    std::shared_ptr<RecordFunctor>& rec) {
  fusion_cache_ptr_->record_hash_map[rec] =
      std::make_unique<FusionCacheEntry>(rec);
}
void FusionManager::createTerminalFusionCacheEntry(
    std::shared_ptr<RecordFunctor>& rec) {
  ++num_fusions_;
  TORCH_CHECK(num_fusions_ <= max_fusions_,
      "The number of fusions in nvfuser has exceeded ", max_fusions_,
      "fusions.  The max_fusions for the FusionManager might need to be ",
      "increased if the max number is not being exceeded due to an error.");
  fusion_cache_ptr_->record_hash_map[rec] =
      std::make_unique<FusionCacheEntry>();
}
void FusionManager::resetFusionCachePtr() {
  fusion_cache_ptr_ = fusion_cache_start_.get();
}
void FusionManager::traverseFusionCache(std::shared_ptr<RecordFunctor>& rec) {
  fusion_cache_ptr_ = fusionCachePtr()->record_hash_map[rec].get();
}

FusionCacheEntry* FusionManager::fusionCachePtr() const {
  TORCH_INTERNAL_ASSERT(fusion_cache_ptr_ != nullptr,
      "The fusion cache entry is unexpectedly null.");
  return fusion_cache_ptr_;
}
Nvf::FusionExecutorCache* FusionManager::fusionExecutorCachePtr() const {
  if (fusionCachePtr()->fusion_executor_cache) {
    return fusionCachePtr()->fusion_executor_cache.get();
  } else {
    TORCH_INTERNAL_ASSERT(false, "The Fusion Executor Cache Pointer is Null");
    return nullptr;
  }
}
Nvf::Fusion* FusionManager::fusionPtr() const {
  TORCH_INTERNAL_ASSERT(fusionExecutorCachePtr()->fusion() != nullptr,
      "The fusion pointer is null.");
  return fusionExecutorCachePtr()->fusion();
}

} // namespace nvfuser

#endif // USE_CUDA
