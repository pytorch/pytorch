#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <deque>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

class TensorView;

class ComputeAt {
 public:
  static void run(
      TensorView* _producer,
      TensorView* _consumer,
      unsigned int _consumer_position);

 private:
  TensorView* producer_;
  TensorView* consumer_;
  unsigned int consumer_position_;

  // Only keeping these as member functions as ComputeAt is friend of TensorView
  // Don't want to keep expanding things that are friends of TV.
  // Runs replayPasC and sets producer computeAt settings
  void computeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_compute_at_axis);

  // Runs replay, and checks computeAt position of producer. If new position
  // would be higher, actually runs operation.
  void maybe_computeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_compute_at_axis);

  // Runs replayCasP and sets producer computeAt settings
  void forwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_compute_at_axis);

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  void setCommonConsumer();

  // Propagate in a way we know result will be correct, which is forward from
  // consumer and backward from consumer to producer
  void traverseAllKnown();

  // Traverse from producer to common_consumer if exists or through all uses of
  // producer
  void traverseForward();

  // Propagate backward from consumer or common consumer, check if it increase
  // computeAt position on tensors, if so take it!
  void traverseBackward();

  // Run the computeAt pass
  void runPass();

  // Set outputs relative to eachother if there is not a common consumer
  void setupOutputs();

  // Common consumer if it exists
  TensorView* common_consumer_ = nullptr;

  // Producer use chains set in, used in a few spots.
  std::deque<std::deque<TensorView*>> producer_use_chains_;

  // Order for forward computeAt pass
  std::vector<std::pair<TensorView*, TensorView*>> forward_compute_at_order;

  // Order for backward computeAt pass
  std::vector<std::pair<TensorView*, TensorView*>> backward_compute_at_order;

  // TensorViews we've set computeAt of, in this computeAt pass
  std::unordered_set<TensorView*> compute_at_ed;

  // TensorViews of which we know their correct computeAt position
  std::unordered_map<TensorView*, unsigned int> known_positions;

  ComputeAt(
      TensorView* _producer,
      TensorView* _consumer,
      unsigned int _consumer_position);
};

} // namespace fuser
} // namespace jit
} // namespace torch
