// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

namespace c10d::nccl2 {

WorkNCCL::WorkStatus WorkNCCLQueue::garbageCollectLocked() {
  WorkNCCL::WorkStatus last_status = WorkNCCL::WorkStatus::COMPLETED;

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
      WorkNCCL::WorkStatus status = work->checkStatus();

      if (status == WorkNCCL::WorkStatus::COMPLETED) {
        // Work is completed, remove it from the work queue
        work_queue.pop();
        // Continue to the next element in the queue
      } else if (
          status == WorkNCCL::WorkStatus::TIMEDOUT ||
          status == WorkNCCL::WorkStatus::ERROR) {
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
WorkNCCL::WorkStatus WorkNCCLQueue::garbageCollect() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return garbageCollectLocked();
}

WorkNCCL::WorkStatus WorkNCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::lock_guard<std::mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  WorkNCCL::WorkStatus status = WorkNCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollectLocked();
    if (status == WorkNCCL::WorkStatus::ERROR ||
        status == WorkNCCL::WorkStatus::TIMEDOUT ||
        status == WorkNCCL::WorkStatus::COMPLETED) {
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

void WorkNCCLQueue::enqueueWork(
    c10::intrusive_ptr<WorkNCCL> work,
    cudaStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(std::move(work));
}

} // namespace c10d::nccl2
