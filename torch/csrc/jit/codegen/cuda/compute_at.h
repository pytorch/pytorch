#pragma once

#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <deque>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TensorDomain;
class TensorView;

// We're going to keep data related to the computeAt pass for each TensorView in
// this structure, this will allow us to keep a single entry in a map from a
// TensorView to this one.
class ComputeAtData {
 public:
  ComputeAtData() = default;
  ComputeAtData(TensorView* tv);

  // Clear after a given traversal. There will be more than one.
  void clearPass();

  // Makes sure value matches current_traversal_position if
  // current_traversal_position_set is true. If this is not the case we're in
  // an invalid compute_at that would require tensor replication.
  void setPassPosition(unsigned int pos);

  unsigned int getPassPosition() {
    return current_traversal_position;
  }

 private:
  // Hold onto the provided TensorView, only used for error message
  TensorView* tv_ref_ = nullptr;

  // What was the computeAt position before the computeAt pass started
  unsigned int original_compute_at_position = 0;

  // Position we can update during a traversal
  unsigned int current_traversal_position = 0;

  // Did this traversal set a position or not yet
  bool current_traversal_position_set = false;
};

class ComputeAt {
 public:
  // Runs the compute at pass making producer look like consumer, computing
  // producer relative to consumer
  static void runAt(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_position);

  // Runs the compute with pass making consumer look like producer, computing
  // producer relative to consumer
  static void runWith(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_position);

 private:
  TensorView* producer_;
  TensorView* consumer_;
  TensorView* reference_;
  unsigned int reference_position_;
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

  // Propagate backward from consumer to producer, check if it increase
  // computeAt position on tensors, if so take it!
  void traverseBackward();

  // Traverse from producer to common_consumer if it exists or through all uses
  // of producer
  void traverseForward();

  // Run the computeAt pass
  void runPass();

  // Common consumer if it exists
  TensorView* common_consumer_ = nullptr;

  // Producer use chains set in, used in a few spots.
  std::deque<std::deque<TensorView*>> producer_use_chains_;

  // All we need to know and keep track of for each TensorView in this pass.
  std::unordered_map<TensorView*, ComputeAtData> tv_data;

  ComputeAt(
      TensorView* _producer,
      TensorView* _consumer,
      TensorView* _reference,
      unsigned int _reference_position);

  ComputeAt() = delete;
  ~ComputeAt() = default;
  ComputeAt(ComputeAt&) = delete;
  ComputeAt& operator=(const ComputeAt& other) = delete;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
