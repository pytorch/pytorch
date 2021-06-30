#pragma once

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

  // Returns if new postion is greater or equal to previous seen, if
  bool shouldSetComputeAt(unsigned int pos) const {
    return pos > original_compute_at_position &&
        pos > new_compute_at_position && pos >= current_traversal_position;
  }

  // Will return new_compute_at_position, after making sure we cleared out the
  // last pass
  unsigned int getNewPosition() const;

  // Will make sure we haven't invalidated previous computeAt calls by
  // checking that any axes previously in computeAt are still there.
  void validateNewComputeAt() const;

  // Did we ever compute a value for this TV?
  bool touched() const {
    return touched_;
  }

  TensorDomain* getOriginalDomain() const {
    return original_domain_;
  }

  // If we set computeAt, save the domain so we can reset it after traversal.
  // Traversal state can deviate from the domain we will want to save after the
  // entire computeAt pass.
  void setComputeAtDomain(TensorDomain* td);

  // Return domain set in setComputeAtDomain
  TensorDomain* getComputeAtDomain() const {
    return new_compute_at_domain_;
  }

 private:
  // Was the position ever modified?
  bool touched_ = false;

  // Hold onto the provided TensorView
  TensorView* tv_ref_ = nullptr;

  // Did this tv have computeAt set before calling this computeAt pass?
  bool original_has_compute_at_ = false;

  // What was the computeAt position before the computeAt pass started
  unsigned int original_compute_at_position = 0;

  // and what was the previous domain that position was set relative to.
  TensorDomain* original_domain_ = nullptr;

  // Position we can update during a traversal
  unsigned int current_traversal_position = 0;

  // Did this traversal set a position or not yet
  bool current_traversal_position_set = false;

  // Position to update after a traversal
  unsigned int new_compute_at_position = 0;

  // Domain when we actually set computeAt, will set back to this after the
  // pass.
  TensorDomain* new_compute_at_domain_;
};

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

  // Runs replayPasC and sets producer computeAt settings. Returns
  // producer_compute_at_axis.
  unsigned int backwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_compute_at_axis);

  // Runs replayCasP and sets producer computeAt settings. Returns
  // consumer_compute_at_axis.
  unsigned int forwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_compute_at_axis);

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

  // Set outputs relative to eachother if there is not a common consumer
  void setupOutputs();

  // Common consumer if it exists
  TensorView* common_consumer_ = nullptr;

  // Producer use chains set in, used in a few spots.
  std::deque<std::deque<TensorView*>> producer_use_chains_;

  // All we need to know and keep track of for each TensorView in this pass.
  std::unordered_map<TensorView*, ComputeAtData> tv_data;

  ComputeAt(
      TensorView* _producer,
      TensorView* _consumer,
      unsigned int _consumer_position);

  ComputeAt() = delete;
  ~ComputeAt() = default;
  ComputeAt(ComputeAt&) = delete;
  ComputeAt& operator=(const ComputeAt& other) = delete;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
