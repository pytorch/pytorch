#pragma once
#include <memory>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_owner.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

namespace nvfuser {

class FusionCacheEntry {
 private:
  bool is_end;
  std::unique_ptr<FusionOwner> fusion_owner_;

  std::shared_ptr<RecordFunctor> record_;
  std::unordered_map<RecordFunctor*, std::unique_ptr<FusionCacheEntry>>
      record_hash_map_;
};

class FusionManager {

  c10::optional<FusionCacheEntry*> lookupFusionCacheEntry(RecordFunctor* rec);
  void createFusionCacheEntry(std::shared_ptr<RecordFunctor> &rec);

 private:
  //! The fusion cache is implemented as a prefix tree of entries containing
  //! a Record representing a Fusion Definition line entry.
  std::unique_ptr<FusionCacheEntry> fusion_cache_start_ptr_;
  FusionCacheEntry* fuson_cache_current_ptr_;
};

} // namespace nvfuser
