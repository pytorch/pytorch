#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/maxinfo_propagator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Simple selector that only propagates across tensor views in the provided
// unordered_set. Will also propagate to all consumers of those tensors, and the
// siblings of those tensors.
class InlinePropagatorSelector : public MaxInfoSpanningTree::Selector {
  std::unordered_set<TensorView*> selected_;

 public:
  virtual bool allowC2P(TensorView* from, TensorView* to) override;
  virtual bool allowP2C(TensorView* from, TensorView* to) override;
  virtual bool allowSibling(TensorView* from, TensorView* to) override;

  InlinePropagatorSelector(std::unordered_set<TensorView*> selected)
      : selected_(std::move(selected)){};
  const std::unordered_set<TensorView*>& selected() const {
    return selected_;
  }
};

class MaxPosCalculator {
  ComputeAtMode mode_ = ComputeAtMode::Standard;

  // Root domains in producer that's unmappable to any of its consumers
  std::unordered_set<IterDomain*> unmappable_dims_;

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
  size_t getMaxPosC2P(TensorView* from, TensorView* to) const;

  // Returns the maximum position consumer can be inlined based on producer
  // given the set ComputeAtMode
  size_t getMaxPosP2C(TensorView* from, TensorView* to) const;

  MaxPosCalculator(ComputeAtMode mode);
};

class InlinePropagator : public MaxInfoSpanningTree::Propagator {
  // Checks producers and consumers to see what the maximum position in tv is
  // that can be shared across both directions.
  size_t getMaxPosAll(TensorView* tv);

  // Returns the inline position in consumer that producer should be inlined as
  // based on consumer, taking into consideration the max possible returned by
  // getMaxPos{P2C, C2P}, the compute at mode type.
  size_t getFromPosC2P(TensorView* from, TensorView* to);

  // Returns the inline position in producer that consumer should be inlined as
  // based on producer, taking into consideration the max possible returned by
  // getMaxPos{P2C, C2P}, the compute at mode type.
  size_t getFromPosP2C(TensorView* from, TensorView* to);

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
  void setCAPos(TensorView* tv, size_t pos);

  const MaxPosCalculator max_pos_calc;
  std::unordered_set<TensorView*> selected_;
  TensorView* reference_;
  size_t reference_pos_;
  ComputeAtMode mode_ = ComputeAtMode::Standard;
  bool is_first_ = true;

 public:
  InlinePropagator(
      std::unordered_set<TensorView*> selected,
      TensorView* reference,
      int64_t reference_pos,
      ComputeAtMode mode);

  ~InlinePropagator() = default;

  // Actually propagate the transformations for the inlining pass. Uses the
  // functions above to figure out what position to do the propagation at.
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
};

// This is actually not a propagation, it only sets the max producer position of
// the tensors, and it is not needed to compute the max producer position in a
// specific order. But MaxInfoSpanningTree provides a very convenient API to
// visit the tensors, so I just use it for cleaner code.
class MaxProducerPosUpdater : public MaxInfoSpanningTree::Propagator {
  std::unordered_set<TensorView*> updated_;
  void handle(TensorView* tv);

 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
