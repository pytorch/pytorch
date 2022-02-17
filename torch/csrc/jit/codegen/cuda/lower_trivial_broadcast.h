#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Traverse and collect all concretized broadcast domains.
//!
//! The traversal first initializes the origin map with broadcast
//! domains in input tensors. Then, a new entry is added to the origin
//! map when a broadcast op is encountered during a forward traversal
//! of the given fusion. For non-broadcast ops, mappings are just
//! propagated forward using PairwiseRootDomainMap.
//!
//! When the mapped consumer domain is not broadcast, it means the
//! producer broadcast domain is concretized, and its origin broadcast
//! domains are marked as concretized.
class TORCH_CUDA_CU_API ConcretizedBroadcastDomains : private IterVisitor {
 public:
  void build(Fusion* fusion);

  bool isConcretized(IterDomain* id) const;

 private:
  using IterVisitor::handle;

  void handle(BroadcastOp* bop) final;

  void handle(Expr* expr) final;

  void markAsConcretized(IterDomain* root_domain);

 private:
  //! Maps each broadcast domain to its original broadcast
  //! domains. Their can be multiple original domains due to, e.g.,
  //! binary ops with broadcast domains in both inputs.
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      broadcast_origin_map_;
  //! Set of all concretized original domains
  std::unordered_set<IterDomain*> concretized_domains_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
