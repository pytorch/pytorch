#pragma once

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace pointwise_utils {

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all IterDomains in the fusion.
class DomainMap {
 public:
  DomainMap(Fusion* fusion);
  virtual ~DomainMap() = default;

  bool areExactMapped(IterDomain* id1, IterDomain* id2);

  const ComputeAtMap& getComputeAtMap() const {
    return ca_map_;
  }

 protected:
  // Determine if all iterDomains are mapped between input and output tvs
  bool areAllInputIdsMappedToOutput(TensorView* input_tv, TensorView* output_tv)
      const;

  // Erase input concrete ID if it is mapped to output ID
  bool eraseIfMapped(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const;

  // Check if in_id is mapped to out_id through any view rfactor domain
  void eraseIfInputMappedThroughViewToOutput(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const;

  // Find any id in domain that maps with target id
  IterDomain* anyMapped(
      const std::vector<IterDomain*>& domain,
      IterDomain* target) const;

  Fusion* fusion_ = nullptr;
  ComputeAtMap ca_map_;
  std::vector<TensorView*> view_tvs_;
};

// Returns number of non-reduction/non-broadcast dims in rfactor domain
inline size_t nRootDims(const TensorView* tv) {
  auto root_dom = tv->getMaybeRFactorDomain();
  size_t tv_n_dims = 0;
  for (auto dim : root_dom) {
    if (!dim->isReduction() && !dim->isBroadcast()) {
      tv_n_dims++;
    }
  }
  return tv_n_dims;
}

} // namespace pointwise_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
