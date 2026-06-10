// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/nccl/TorchWorkNCCL.hpp>

namespace torch::comms {

TorchWorkNCCL::WorkStatus TorchWorkNCCLQueue::garbageCollectLocked() {
  TorchWorkNCCL::WorkStatus last_status = TorchWorkNCCL::WorkStatus::COMPLETED;

  // Keep popping completed elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      // Get the first work object in the queue
      auto work = work_queue.front();

      // Use the checkStatus function to determine the work status
      TorchWorkNCCL::WorkStatus status = work->checkStatus();

      if (status == TorchWorkNCCL::WorkStatus::COMPLETED) {
        // Work is completed, remove it from the work queue
        work_queue.pop();
        // Continue to the next element in the queue
      } else if (
          status == TorchWorkNCCL::WorkStatus::TIMEDOUT ||
          status == TorchWorkNCCL::WorkStatus::ERROR) {
        // Return the error status immediately
        return status;
      } else {
        // NOT_STARTED or INPROGRESS - stop processing this queue
        last_status = status;
        break;
      }
    }

    // If the queue is now empty, remove it from the map
    if (work_queue.empty()) {
      it = stream_work_queues_.erase(it);
    } else {
      ++it;
    }
  }

  return last_status;
}

// Thread-safety: This method is called from the timeout watchdog thread while
// the main thread may be enqueuing work via enqueueWork(). The
// work_queues_mutex_ ensures proper synchronization - both garbageCollect() and
// enqueueWork() acquire the mutex before accessing stream_work_queues_.
TorchWorkNCCL::WorkStatus TorchWorkNCCLQueue::garbageCollect() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return garbageCollectLocked();
}

TorchWorkNCCL::WorkStatus TorchWorkNCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::lock_guard<std::mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  TorchWorkNCCL::WorkStatus status = TorchWorkNCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollectLocked();
    if (status == TorchWorkNCCL::WorkStatus::ERROR ||
        status == TorchWorkNCCL::WorkStatus::TIMEDOUT ||
        status == TorchWorkNCCL::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues & completed work queue.
  //
  // NOTE: finalize MUST return without holding references to any work object,
  // otherwise it may leak object and cause side effects.
  stream_work_queues_.clear();

  return status;
}

void TorchWorkNCCLQueue::enqueueWork(
    c10::intrusive_ptr<TorchWorkNCCL> work,
    cudaStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(std::move(work));
}

} // namespace torch::comms
