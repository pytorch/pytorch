#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/WorkXCCL.hpp>

namespace c10d::xccl2 {

WorkXCCL::WorkStatus WorkXCCLQueue::garbageCollectLocked() {
  WorkXCCL::WorkStatus overall_status = WorkXCCL::WorkStatus::COMPLETED;

  // Keep popping completed elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      // Get the first work object in the queue
      auto& work = work_queue.front();

      // Use the checkStatus function to determine the work status
      WorkXCCL::WorkStatus status = work->checkStatus();

      if (status == WorkXCCL::WorkStatus::COMPLETED) {
        // Tensor references must be released by a caller thread, not by the
        // watchdog that runs garbageCollect(), so park the work here and let
        // the next enqueueWork() drop it.
        completed_work_queue_.push(std::move(work));
        work_queue.pop();
        // Continue to the next element in the queue
      } else if (
          status == WorkXCCL::WorkStatus::TIMEDOUT ||
          status == WorkXCCL::WorkStatus::ERROR) {
        // Return the error status immediately
        return status;
      } else {
        // NOT_STARTED or INPROGRESS - stop processing this queue
        if (overall_status == WorkXCCL::WorkStatus::COMPLETED) {
          overall_status = status;
        }
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

  return overall_status;
}

// Thread-safety: This method is called from the timeout watchdog thread while
// the main thread may be enqueuing work via enqueueWork(). The
// work_queues_mutex_ ensures proper synchronization - both garbageCollect() and
// enqueueWork() acquire the mutex before accessing stream_work_queues_.
WorkXCCL::WorkStatus WorkXCCLQueue::garbageCollect() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return garbageCollectLocked();
}

WorkXCCL::WorkStatus WorkXCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::unique_lock<std::mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  WorkXCCL::WorkStatus status = WorkXCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollectLocked();
    if (status == WorkXCCL::WorkStatus::ERROR ||
        status == WorkXCCL::WorkStatus::TIMEDOUT ||
        status == WorkXCCL::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues & completed work queue.
  //
  // NOTE: finalize MUST return without holding references to any work object,
  // otherwise it may leak object and cause side effects.
  stream_work_queues_.clear();
  std::queue<c10::intrusive_ptr<WorkXCCL>> completed_work_queue;
  completed_work_queue.swap(completed_work_queue_);
  lock.unlock();

  return status;
}

bool WorkXCCLQueue::empty() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return stream_work_queues_.empty();
}

void WorkXCCLQueue::enqueueWork(
    c10::intrusive_ptr<WorkXCCL> work,
    c10::xpu::XPUStream stream) {
  // Drop the previously completed work outside the lock, on this (caller)
  // thread.
  std::queue<c10::intrusive_ptr<WorkXCCL>> completed_work_queue;
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    completed_work_queue.swap(completed_work_queue_);
    // Add work to stream's queue after events have been recorded
    stream_work_queues_[stream].push(std::move(work));
  }
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
