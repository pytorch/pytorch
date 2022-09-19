#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Detect almost all IterDomains that are derived from trivial
//! reductons.
class TORCH_CUDA_CU_API TrivialReductionInfo {
 public:
  void build(Fusion* fusion);

  bool isDerived(IterDomain* id) const;

 private:
  //! IterDomains that are derived only from trivial
  //! reductons. Included domains are not limited to reduction axes as
  //! rfactor can make reductions to normal axes.
  //!
  //! Note that the set should cover almost all cases but there can be
  //! undetected trivial domains. For example, split by one creates a
  //! trivial reduction domain, which is detected. However, if it is
  //! further split, both of the two resulting axes are also trivial,
  //! however, only the inner axis is recognized as trivial. While this
  //! is a limitation, it would have very little practical
  //! implication.
  std::unordered_set<IterDomain*> domains_;
  //! Subset of domains_, whose input root axes are all derived from
  //! trivial reductions. These domains do not need to manifest as
  //! for-loops.
  std::unordered_set<IterDomain*> domains_derived_from_root_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
