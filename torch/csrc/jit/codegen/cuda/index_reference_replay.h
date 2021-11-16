#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct ReferenceTensor {
  TensorDomain* domain = nullptr;

  // Map from concrete iteration domains in ComputeAtMaps to iter domains
  // including those used to construct domain.
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_id;
};

class IndexReferenceReplay : public OptInDispatch {
 private:
  IndexReferenceReplay(const std::vector<kir::ForLoop*>& loop_structure)
      : loop_structure_(loop_structure) {}

  // We're going to replay this split operation on the corresponding ID
  void handle(Split* s) override;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(Merge* m) override;

  TensorDomain* computeReplay();

  using OptInDispatch::handle;

 private:
  const std::vector<kir::ForLoop*>& loop_structure_;

  // Replay map
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_id_;

  // Replay map
  std::unordered_set<IterDomain*> leaf_ids_;

 public:
  static ReferenceTensor getReference(
      const std::vector<kir::ForLoop*>& loop_structure) {
    auto replay = IndexReferenceReplay(loop_structure);
    ReferenceTensor ref;
    ref.domain = replay.computeReplay();
    ref.concrete_to_id = replay.concrete_to_id_;
    return ref;
  }
};

// Index into the reference based on the provided index map.
IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_domain,
    std::unordered_map<kir::IterDomain*, kir::Val*> index_map,
    std::unordered_set<IterDomain*> preferred_path,
    std::unordered_map<kir::IterDomain*, kir::Val*> halo_extent_map = {});

// Short cut for global TVs. Index into the reference based on all loop indicies
// in the loop structure.
IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_domain);

// When indexing there are sometimes an option to propagate an index down
// multiple paths. This will return the IterDomains in the history of the
// reference domain and mark which paths should be taken (if there's a
// preference) to reach the roots provided in preferred_roots.
std::unordered_set<IterDomain*> buildPreferredPaths(
    TensorDomain* reference_domain,
    const std::unordered_set<IterDomain*>& preferred_roots);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
