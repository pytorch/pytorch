#pragma once

#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TensorDomain;
class TensorView;

class ComputeAt {
 public:
  // Runs the compute at pass making producer look like consumer, computing
  // producer relative to consumer
  static void runAt(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  // Runs the compute with pass making consumer look like producer, computing
  // producer relative to consumer
  static void runWith(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  ComputeAt() = delete;
  ComputeAt(ComputeAt&) = delete;
  ComputeAt& operator=(const ComputeAt& other) = delete;

 private:
  TensorView* producer_;
  TensorView* consumer_;
  TensorView* reference_;
  unsigned int reference_position_;
  ComputeAtMode mode_ = ComputeAtMode::Standard;

  unsigned int producer_position_ = 0;
  ComputeAtRootDomainMap root_map_;

  // Runs replayPasC and sets producer computeAt settings. Returns
  // producer_compute_at_pos.
  unsigned int backwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_compute_at_pos);

  // Runs replayCasP and sets producer computeAt settings. Returns
  // consumer_compute_at_pos.
  unsigned int forwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_compute_at_pos);

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  void setCommonConsumer();

  // Iterate through all TVs and collect the dimensions of each TV that don't
  // map to all its consumer TVs.
  void buildUnmappableDims();

  // Propagate backward from consumer to producer, check if it increase
  // computeAt position on tensors, if so take it!
  void traverseBackward();

  // Traverse from producer to common_consumer if it exists or through all uses
  // of producer
  void traverseForward();

  // Looks at producer tensor views of consumer_tv, recomputes its max
  // producer position, and sets max producer position. This function can
  // only potentially lower the max producer position of consumer_tv.
  void resetMaxProducerPos(TensorView* consumer_tv);

  // Undo the inlining of block broadcast at the innermost positions
  //  to avoid generating repeated block broadcasts
  void hoistInnermostBroadcast();

  // Update multi-output expressions. If one output is modified, all outputs
  // should be modified as well. Propagate transformations, compute at, and
  // produce at from tv to siblings. Run as final pass as it will invalidate the
  // computeAt map originally computed.
  void updateSiblings();

  // Compute at pass requires tracking "maxProducerPosition" even if set simply
  // from input tensor views. However, when lowering, we need a valid produce at
  // position of all tensors, so inputs should never actually set their
  // consumers maxProduceAt position.
  void updateInputProduceAts();

  // Run the computeAt pass
  void runPass();

  // Common consumer if it exists
  TensorView* common_consumer_ = nullptr;

  // Producer use chains set in, used in a few spots.
  std::deque<std::deque<TensorView*>> producer_use_chains_;

  // Root domains in producer that's unmappable to any of its consumers
  std::unordered_set<IterDomain*> unmappable_dims_;

  ComputeAt(
      TensorView* _producer,
      TensorView* _consumer,
      TensorView* _reference,
      unsigned int _reference_position,
      ComputeAtMode _mode);

  ~ComputeAt() = default;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
