#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class GpuLower;

//! Detect almost all IterDomains that are derived from trivial
//! reductons.
class TORCH_CUDA_CU_API TrivialReductionInfo {
 public:
  void build(Fusion* fusion, GpuLower* gpu_lower);

  bool isDerived(IterDomain* id) const;
  bool isDerivedFromRoot(IterDomain* id) const;

  bool isDerived(kir::IterDomain* id) const;
  bool isDerivedFromRoot(kir::IterDomain* id) const;

 private:
  //! Convert the sets to KIR sets
  void buildKir(Fusion* fusion, GpuLower* gpu_lower);

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

  std::unordered_set<kir::IterDomain*> kir_domains_;
  std::unordered_set<kir::IterDomain*> kir_domains_derived_from_root_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
