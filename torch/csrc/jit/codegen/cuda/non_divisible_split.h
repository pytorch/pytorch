#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! If an IterDomain is split and its inner output domain is
//! eventually split too, the second split must be divisible or the
//! inner domain must be predicated. This class finds Split
//! expressions that need to be divisible or predicated.
//!
//! Second splits are not limited to just direct output domains of
//! first splits but also indirect descendent domains as well.
//!
//! Predicating non-divisible split domains does not work if split
//! output domains are vectorized where ParallelType::Vectorize is
//! applied to an inner domain of splits. If it's non-divisible,
//! predicating the input domain of the non-divisible split results in
//! a vectoried operation is predicated out entirely since we do not
//! generate a fall-back non-vectorized else path. Runtime check is
//! done for those domains.
class TORCH_CUDA_CU_API NonDivisibleSplitInfo : public IterVisitor {
 public:
  void build(Fusion* fusion);

  const auto& splitsToPredicate() const {
    return splits_to_predicate_;
  }

  const auto& splitsToValidate() const {
    return splits_to_validate_;
  }

 private:
  using IterVisitor::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  //! True if reachable from inner domains of splits
  bool isReachableFromInnerDomains(IterDomain* id) const;

  //! Forward propagate the reachability information
  void propagateReachability(Split* split, bool is_protected);

  //! Forward propagate the reachability information
  void propagateReachability(Merge* merge);

  void clearReachability();

  //! Returns the extent of a split output domain if it's not proven to
  //! be divisible.
  Val* getMaybeNonDivisibleExtent(Split* split) const;

  //! Remove redundant predicates as divisibility may be validated at
  //! run time
  void removeRedundancy();

 private:
  //! Split expressions whose input domain must be predicated
  std::unordered_map<TensorView*, std::vector<Split*>> splits_to_predicate_;
  //! Split expressions whose divisibility must be validated at run time
  std::unordered_set<Split*> splits_to_validate_;

  //! Temporarily used for analyzing each tensor
  TensorView* current_tv_ = nullptr;
  std::unordered_set<IterDomain*> inner_domains_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
