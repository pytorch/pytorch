#pragma once
#include <memory>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

namespace nvfuser {

class FusionCacheEntry {
 private:
  std::unique_ptr<RecordFunctor> record_;
  std::unordered_map<RecordFunctor*, std::unique_ptr<FusionCacheEntry>>
      record_hash_map_;
};

class FusionManager {
 private:
  //! The fusion cache is implemented as a prefix tree of entries containing
  //! a Record representing a Fusion Definition line entry.
  std::unique_ptr<FusionCacheEntry> fusion_cache_;
};

} // namespace nvfuser
