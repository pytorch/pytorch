#include "caffe2/core/net_async_scheduling.h"

namespace caffe2 {

AsyncSchedulingNet::AsyncSchedulingNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : AsyncNetBase(net_def, ws), running_(false) {}

void AsyncSchedulingNet::reset() {
  AsyncNetBase::reset();

  processed_tasks_num_ = 0;
  success_ = true;

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto& task_ops = chains_[task_id];
    auto& task_op_node = operator_nodes_[task_ops.front()];
    task_op_node.runtime_parent_count_ = parents(task_id).size();
  }
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
      int stream_id = 0;
      if (streams_per_gpu_ > 1) {
        stream_id = stream(task_id);
      }
      try {
        run(task_id, stream_id);
      } catch (const std::exception& e) {
        success_ = false;
      }
    }

    for (auto child_id : children(task_id)) {
      int parent_count = updateParentCount(child_id);
      if (parent_count == 0) {
        // Schedule a child if:
        // - there is failure, we skip an op execution and finish the job
        // - forced scheduling though --caffe2_net_async_always_schedule_child
        // - --caffe2_net_async_finish_chain is set, in this case parents are
        //   guaranteed to be finished
        // - in all other cases, check parents with canSchedule
        if (!success_ || always_schedule_child_ || finish_chain_ ||
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

    // finishRun may cause waiters to wake up and destroy the net,
    // before we call finishRun we need to make sure all other (finishing)
    // tasks are done;
    // Bumping and checking the counter after the task's job is done
    auto tasks_num = tasksNum();
    auto cur_processed_tasks = ++processed_tasks_num_;
    if (cur_processed_tasks == tasks_num) {
      finishRun();
    }
  });
}

void AsyncSchedulingNet::pollAndSchedule(int task_id) {
  bool parent_failed = false;
  bool can_schedule = canSchedule(task_id, nullptr, &parent_failed);
  if (parent_failed) {
    success_ = false;
  }
  // schedule the task if:
  //  - parents are ready
  //  - we failed / cleanup started (no ops will run)

  if (can_schedule || !success_ || parent_failed) {
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
  {
    std::unique_lock<std::mutex> lock(running_mutex_);
    running_ = false;
  }
  // wait for scheduled ops and make sure all events are marked as finished
  finalizeEvents();
  // notify observers and waiters
  StopAllObservers();
  running_cv_.notify_all();
}

bool AsyncSchedulingNet::RunAsync() {
  try {
    std::unique_lock<std::mutex> lock(running_mutex_);
    if (running_) {
      LOG(ERROR) << "Detected concurrent runs";
      return false;
    }
    running_ = true;
    reset();

    StartAllObservers();

    for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
      if (parents(task_id).empty()) {
        schedule(task_id);
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception while starting an async run: " << e.what();
    finishRun();
    return false;
  }

  if (tasksNum() == 0) {
    finishRun();
  }

  return true;
}

AsyncSchedulingNet::~AsyncSchedulingNet() {
  Wait();
}

REGISTER_NET(async_scheduling, AsyncSchedulingNet);

} // namespace caffe2
