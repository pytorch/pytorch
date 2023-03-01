#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Collects start and stop offsets of all split root domains. Offsets
//! are zero unless partially split.
class TORCH_CUDA_CU_API PartialSplitMap {
 public:
  void build(Fusion* fusion);

  Val* getStartOffset(IterDomain* root_domain) const;
  Val* getStopOffset(IterDomain* root_domain) const;

 private:
  std::unordered_map<IterDomain*, Val*> start_offset_map_;
  std::unordered_map<IterDomain*, Val*> stop_offset_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
