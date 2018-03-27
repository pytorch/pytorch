#include "caffe2/core/net_async_scheduling.h"

CAFFE2_DEFINE_bool(
    caffe2_net_async_always_schedule_child,
    false,
    "Always schedule child chains from parent chain");

namespace caffe2 {

AsyncSchedulingNet::AsyncSchedulingNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : AsyncNetBase(net_def, ws), running_(false) {
  reset();
}

void AsyncSchedulingNet::reset() {
  processed_tasks_num_ = 0;
  cleanup_ = false;
  success_ = true;

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto& task_ops = chains_[task_id];
    auto& task_op_node = operator_nodes_[task_ops.front()];
    task_op_node.runtime_parent_count_ = parents(task_id).size();
  }
  exception_messages_.clear();
}

void AsyncSchedulingNet::Wait() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  while (running_) {
    running_cv_.wait(lock);
  }
}

void AsyncSchedulingNet::schedule(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  pool(device_option)->run([this, task_id]() {
    if (success_) {
      int stream_id = stream(task_id);
      asyncWait(task_id, stream_id, parents(task_id));
      try {
        run(task_id, stream_id);
      } catch (const std::exception& e) {
        std::unique_lock<std::mutex> lock(exception_mutex_);
        exception_messages_.push_back(e.what());
        success_ = false;
      }
    }

    auto task_count = ++processed_tasks_num_;

    for (auto child_id : children(task_id)) {
      int parent_count = updateParentCount(child_id);
      if (parent_count == 0) {
        if (cleanup_ || FLAGS_caffe2_net_async_always_schedule_child ||
            canSchedule(child_id)) {
          schedule(child_id);
        } else {
          const auto& device_option = event(child_id).GetDeviceOption();
          pool(device_option)
              ->run(std::bind(
                  &AsyncSchedulingNet::pollAndSchedule, this, child_id));
        }
      }
    }

    if (success_) {
      if (task_count == tasksNum()) {
        // All tasks are finished, polling thread is sleeping;
        // only one thread enters here
        finalizeEvents();
        finishRun();
        return;
      }
    } else {
      // Before setting running_ to false and notifying waiters we need to
      // 1. Ensure that only one thread does the cleanup
      // 2. Ensure that all other pending tasks in workers and polling threads
      //    are finished and
      // 3. Ensure that all tasks that were not scheduled have their events set
      {
        std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex_);
        if (cleanup_) {
          return;
        }
        cleanup_ = true;
      }

      // Errors are not recoverable and happen in exceptional cases,
      // ok to busy wait
      while (processed_tasks_num_ != tasksNum()) {
      }

      // Make sure all events are set, wait for scheduled events
      finalizeEvents();

      // Notify observers and waiters
      finishRun();
    }
  });
}

void AsyncSchedulingNet::pollAndSchedule(int task_id) {
  if (canSchedule(task_id) || cleanup_) {
    // force schedule the rest of the tasks if cleanup is started
    schedule(task_id);
  } else {
    const auto& device_option = event(task_id).GetDeviceOption();
    pool(device_option)
        ->run(std::bind(&AsyncSchedulingNet::pollAndSchedule, this, task_id));
  }
}

int AsyncSchedulingNet::updateParentCount(int child_id) {
  auto& child_ops = chains_[child_id];
  auto& child_node = operator_nodes_[child_ops.front()];
  int parent_count = --child_node.runtime_parent_count_;
  CAFFE_ENFORCE_GE(parent_count, 0);
  return parent_count;
}

void AsyncSchedulingNet::finishRun() {
  // notify observers and waiters
  StopAllObservers();
  running_ = false;
  running_cv_.notify_all();
}

bool AsyncSchedulingNet::DoRunAsync() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  CAFFE_ENFORCE(!running_, "Concurrent RunAsync calls");
  running_ = true;
  reset();

  StartAllObservers();

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    if (parents(task_id).empty()) {
      schedule(task_id);
    }
  }

  return true;
}

AsyncSchedulingNet::~AsyncSchedulingNet() {}

REGISTER_NET(async_scheduling, AsyncSchedulingNet);

} // namespace caffe2
