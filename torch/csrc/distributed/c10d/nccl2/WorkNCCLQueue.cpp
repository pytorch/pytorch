// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

namespace c10d::nccl2 {

WorkNCCL::WorkStatus WorkNCCLQueue::garbageCollectLocked(
    std::vector<c10::intrusive_ptr<WorkNCCL>>& completed) {
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
        // Report completion after dropping the lock. The queued work owns no
        // result tensors; keep its shared input shelf for caller-thread
        // clearing after completion.
        completed.push_back(work);
        completed_input_tensors_.push(work->inputTensors());
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
  std::vector<c10::intrusive_ptr<WorkNCCL>> completed;
  WorkNCCL::WorkStatus status = WorkNCCL::WorkStatus::COMPLETED;
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    status = garbageCollectLocked(completed);
  }
  // Reported with no queue lock held on purpose: a completion hook may take a
  // lock of its own (c10d::FlightRecorderHook takes the recorder's, which a
  // concurrent dump can hold while it waits on the GIL), and holding
  // work_queues_mutex_ across it would put enqueueWork -- every collective on
  // this backend -- behind that wait.
  for (const auto& work : completed) {
    work->notifyCompletion();
  }
  return status;
}

WorkNCCL::WorkStatus WorkNCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::unique_lock<std::mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  std::vector<c10::intrusive_ptr<WorkNCCL>> completed;
  WorkNCCL::WorkStatus status = WorkNCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollectLocked(completed);
    if (status == WorkNCCL::WorkStatus::ERROR ||
        status == WorkNCCL::WorkStatus::TIMEDOUT ||
        status == WorkNCCL::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues and input tensors.
  //
  // NOTE: finalize MUST return without holding references to any work object,
  // otherwise it may leak object and cause side effects.
  stream_work_queues_.clear();
  std::queue<std::shared_ptr<WorkNCCL::InputTensorShelf>>
      completed_input_tensors;
  completed_input_tensors.swap(completed_input_tensors_);
  lock.unlock();

  for (const auto& work : completed) {
    work->notifyCompletion();
  }
  while (!completed_input_tensors.empty()) {
    completed_input_tensors.front()->clear();
    completed_input_tensors.pop();
  }
  return status;
}

void WorkNCCLQueue::enqueueWork(
    const c10::intrusive_ptr<WorkNCCL>& work,
    cudaStream_t stream) {
  std::queue<std::shared_ptr<WorkNCCL::InputTensorShelf>>
      completed_input_tensors;
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    completed_input_tensors.swap(completed_input_tensors_);
    stream_work_queues_[stream].push(work->createTrackingWork());
  }
  while (!completed_input_tensors.empty()) {
    completed_input_tensors.front()->clear();
    completed_input_tensors.pop();
  }
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
