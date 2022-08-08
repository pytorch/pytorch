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

FusionManager::FusionManager()
    : start_record_(new StartRecord()),
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
  auto cache_entry = fusion_cache_ptr_->record_hash_map.find(rec);
  if (cache_entry == std::end(fusion_cache_ptr_->record_hash_map)) {
    return c10::nullopt;
  } else {
    return c10::optional<FusionCacheEntry*>(cache_entry->second.get());
  }
}
void FusionManager::createFusionCacheEntry(
    std::shared_ptr<RecordFunctor>& rec) {
  fusion_cache_ptr_->record_hash_map[rec] =
      std::make_unique<FusionCacheEntry>(rec);
}
void FusionManager::createTerminalFusionCacheEntry(
    std::shared_ptr<RecordFunctor>& rec) {
  fusion_cache_ptr_->record_hash_map[rec] =
      std::make_unique<FusionCacheEntry>();
}
void FusionManager::resetFusionCachePtr() {
  fusion_cache_ptr_ = fusion_cache_start_.get();
}
void FusionManager::traverseFusionCache(std::shared_ptr<RecordFunctor>& rec) {
  fusion_cache_ptr_ = fusion_cache_ptr_->record_hash_map[rec].get();
}

Nvf::FusionExecutorCache* FusionManager::fusionExecutorCachePtr() const {
  //! \todo add pointer checks
  return fusion_cache_ptr_->fusion_executor_cache.get();
}
Nvf::Fusion* FusionManager::fusionPtr() const {
  //! \todo add pointer checks
  return fusionExecutorCachePtr()->fusion();
}

} // namespace nvfuser

#endif // USE_CUDA
