#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Keep track of certain patterns of reductions.
//!
//! - Allreduce IterDomain: reduced and broadcast domain.
class FusedReductionInfo {
 public:
  void markAsAllreduce(IterDomain* id);

  bool isAllreduce(IterDomain* id) const;

 private:
  // Reduction IterDomains that are also broadcast
  std::unordered_set<IterDomain*> allreduce_ids_;
};

//! Detect reductions and broadcasts that are eligible for the fused
//! reduction kernel. When found, the predicate flags of the broadcast
//! is unset, which effectively makes the broadcast just a unary set
//! op.
//! TODO: Consider moving the warp-based fused reduction here.
void fuseReductionsAndBroadcasts(Fusion*);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
