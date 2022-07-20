#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <c10/macros/Export.h>

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

  //! Is a domain concretized?
  bool isConcretized(IterDomain* id) const;

  //! Is a domain concretized to a unique concrete domain?
  bool isUniquelyConcretized(IterDomain* id) const;

  //! Is a domain concretized to multiple concrete domains?
  bool maybeNonUniquelyConcretized(IterDomain* id) const;

 private:
  using IterVisitor::handle;

  void handle(BroadcastOp* bop) final;

  void handle(Expr* expr) final;

  void markAsConcretized(
      IterDomain* broadcast_root_domain,
      IterDomain* concrete_root_domain);

  bool insertRootDomainToConcreteDomainSet(
      IterDomain* new_root_id,
      std::unordered_set<IterDomain*>& id_set);

 private:
  //! Maps each root broadcast domain to its original root broadcast
  //! domains. Their can be multiple original domains due to, e.g.,
  //! binary ops with broadcast domains in both inputs.
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      broadcast_origin_map_;
  //! Map all broadcast domains to concrete root domains
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      broadcast_to_concrete_map_;

  std::unique_ptr<ExactRootDomainMap> exact_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
