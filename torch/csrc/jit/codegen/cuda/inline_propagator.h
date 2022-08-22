#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/maxinfo_propagator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API MaxPosCalculator {
  ComputeAtMode mode_ = ComputeAtMode::Standard;

  // Root domains in producer that's unmappable to any of its consumers
  std::unordered_set<IterDomain*> unmappable_dims_;

  // User set IterDomains to not inline, used in schedulers to avoid inlining
  // trivial reductions
  std::unordered_set<IterDomain*> uninlinable_ids_;

  // Iterate through all TVs and collect the dimensions of each TV that don't
  // map to all its consumer TVs.
  void buildUnmappableDims();

  // Utility function to return if an id of tv is a valid iter domain to inline
  // within. This is used in getMaxPos{PasC,CasP}. Different variations of the
  // bool values are used if checking max position of PasC, CasP, or checking
  // for a max "self" position.
  bool isAllowedID(
      IterDomain* id,
      TensorView* tv,
      bool allow_reduction,
      bool allow_vectorize,
      bool allow_unmappable) const;

 public:
  // Returns the position at which tv can be inlined within.
  size_t getMaxPosSelf(
      TensorView* tv,
      bool allow_reduction,
      bool allow_vectorize,
      bool allow_unmappable) const;

  // Returns the maximum position producer can be inlined based on consumer
  // given the set ComputeAtMode
  size_t getMaxProducerPosFromConsumer(
      TensorView* producer,
      TensorView* consumer) const;

  MaxPosCalculator(
      ComputeAtMode mode,
      std::unordered_set<IterDomain*> uninlinable_ids = {});
};

// Propagate inline position to the `selected` tensors in the DAG. If `selected`
// is not specified or empty, then propagate to the entire DAG.
class TORCH_CUDA_CU_API InlinePropagator
    : public MaxInfoSpanningTree::Propagator {
  // Checks producers and consumers to see what the maximum position in tv is
  // that can be shared across both directions.
  size_t getMaxPosAll(TensorView* tv, bool check_siblings = true);

  // We use mapped_reference_pos_ to keep track of the outer axes information of
  // the reference tensor. That is, mapped_reference_pos_[tv] answers the
  // question "What outer axes in tv are shared with the specified reference
  // tensor's outer axes?". However, when we actually set the CA position of tv,
  // we might not want to set it as mapped_reference_pos_[tv] because because we
  // don't want to inline certain things (such as vectorized dimensions, inner
  // most broadcasting, etc.).
  std::unordered_map<TensorView*, size_t> mapped_reference_pos_;

  // Actually set the computeAt position. This does not necessarily equal to
  // mapped_reference_pos_[tv] because we don't want to inline certain things.
  void setCAPos(TensorView* tv);

  const MaxPosCalculator max_pos_calc;
  std::unordered_set<TensorView*> selected_;
  std::unordered_set<TensorView*> needs_update_max_producer_;
  TensorView* reference_;
  size_t reference_pos_;
  ComputeAtMode mode_ = ComputeAtMode::Standard;

 public:
  InlinePropagator(
      TensorView* reference,
      int64_t reference_pos,
      ComputeAtMode mode = ComputeAtMode::Standard,
      std::unordered_set<TensorView*> selected = {},
      std::unordered_set<IterDomain*> uninlinable_ids = {});

  InlinePropagator(
      TensorView* reference,
      int64_t reference_pos,
      std::unordered_set<TensorView*> selected)
      : InlinePropagator(
            reference,
            reference_pos,
            ComputeAtMode::Standard,
            selected) {}

  ~InlinePropagator() = default;

  // Actually propagate the transformations for the inlining pass. Uses the
  // functions above to figure out what position to do the propagation at.
  virtual void setUp() override;
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
  virtual void tearDown() override;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
