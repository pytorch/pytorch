/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/net_async_polling.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

CAFFE2_DEFINE_int(
    caffe2_streams_per_gpu,
    32,
    "Number of streams per GPU to use in GPU thread pool");

CAFFE2_DECLARE_bool(caffe2_dag_net_collect_stats);

CAFFE2_DEFINE_bool(
    caffe2_net_async_check_stream_status,
    true,
    "Select next non-busy stream");

namespace caffe2 {

AsyncPollingNet::AsyncPollingNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws), running_(false) {
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    operators_.push_back(node.operator_.get());
  }

  const auto& execution_chains = dag_utils::computeChains(operator_nodes_);
  chains_.reserve(execution_chains.size());
  for (const auto& kv : execution_chains) {
    chains_.push_back(kv.second);
  }
  chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

  events_.reserve(chains_.size());
  for (const auto& chain : chains_) {
    const auto& op = operators_[chain.back()];
    events_.push_back(&op->event());
  }

  task_timers_.resize(tasksNum());
  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    task_timers_[task_id] = caffe2::make_unique<Timer>();
  }

  stats_.reserve(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
  for (auto device_idx = 0;
       device_idx < DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
       ++device_idx) {
    stats_.emplace_back(
        "async_polling/stats/" + net_def->name() + "/" +
        caffe2::DeviceTypeName(device_idx));
  }

  reset();
}

void AsyncPollingNet::Wait() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  while (running_) {
    running_cv_.wait(lock);
  }
}

bool AsyncPollingNet::DoRunAsync() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  CAFFE_ENFORCE(!running_, "Concurrent RunAsync calls");
  running_ = true;
  reset();

  StartAllObservers();

  Timer timer;
  pollAndSchedule();
  if (FLAGS_caffe2_dag_net_collect_stats) {
    CAFFE_EVENT(stats_[CPU], poll_time_ms, timer.MilliSeconds());
  }

  StopAllObservers();

  running_ = false;
  running_cv_.notify_all();

  return true;
}

std::shared_ptr<TaskThreadPool> AsyncPollingNet::pool(
    const DeviceOption& device_option) {
  if (!pools_.count(device_option)) {
    std::string device_name = DeviceTypeName(device_option.device_type());
    const auto& pool = ThreadPoolRegistry()->Create(device_name, device_option);
    CAFFE_ENFORCE(pool, "Couldn't get thread pool for device " + device_name);
    pools_[device_option] = pool;
  }
  return pools_[device_option];
}

// Not thread-safe
int AsyncPollingNet::stream(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  int stream_id = 0;
  if (device_option.device_type() == CUDA) {
    int gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE_GE(gpu_id, 0, "Invalid gpu id: " + caffe2::to_string(gpu_id));
    if (gpu_id >= stream_rr_counters_.size()) {
      stream_rr_counters_.resize(gpu_id + 1, 0);
    }
    do {
      stream_id = stream_rr_counters_[gpu_id]++;
      stream_rr_counters_[gpu_id] %= FLAGS_caffe2_streams_per_gpu;
    } while (FLAGS_caffe2_net_async_check_stream_status &&
             !isStreamFree(task_id, stream_id));
  }
  return stream_id;
}

void AsyncPollingNet::schedule(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  int stream_id = stream(task_id);
  if (FLAGS_caffe2_dag_net_collect_stats) {
    task_timers_[task_id]->Start();
  }
  pool(device_option)->run([this, task_id, stream_id, device_option]() {
    if (FLAGS_caffe2_dag_net_collect_stats) {
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_pool_wait_time_us,
          task_timers_[task_id]->MicroSeconds());
    }

    // Non-blocking wait, setups scheduling of dependent async computations;
    // canSchedule ensures that there's no busy wait,
    // for CUDA events we need to insert CUDA event synchronization to ensure
    // that async CUDA computations are executed in correct order
    asyncWait(task_id, stream_id, parents(task_id));
    if (FLAGS_caffe2_dag_net_collect_stats) {
      Timer run_time;
      run(task_id, stream_id);
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_run_time_us,
          run_time.MicroSeconds());
    } else {
      run(task_id, stream_id);
    }
  });
}

bool AsyncPollingNet::canSchedule(int task_id) {
  for (auto parent_id : parents(task_id)) {
    if (!canRunDependency(parent_id, task_id)) {
      return false;
    }
  }

  return true;
}

void AsyncPollingNet::reset() {
  status_.clear();
  status_.resize(tasksNum(), EventStatus::EVENT_INITIALIZED);
  has_chain_failed_ = false;
}

void AsyncPollingNet::pollAndSchedule() {
  std::unordered_set<int> scheduled_tasks;
  std::unordered_set<int> current_tasks;

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    if (parents(task_id).empty()) {
      current_tasks.insert(task_id);
      scheduled_tasks.insert(task_id);
      schedule(task_id);
    }
  }

  Timer timer;
  while (!current_tasks.empty()) {
    std::unordered_set<int> updated_tasks;
    std::unordered_set<int> next_tasks;
    updated_tasks.reserve(current_tasks.size());

    if (FLAGS_caffe2_dag_net_collect_stats) {
      timer.Start();
    }
    for (auto& task_id : current_tasks) {
      if (has_chain_failed_) {
        // error processing is on event->Query()/ErrorMessage() level
        return;
      }
      auto prev_status = status_[task_id];
      status_[task_id] = query(task_id);
      if (status_[task_id] == EventStatus::EVENT_FAILED) {
        return;
      }

      if (prev_status != status_[task_id]) {
        updated_tasks.insert(task_id);
        if (FLAGS_caffe2_dag_net_collect_stats) {
          updateTaskStats(task_id);
        }
      }

      if (status_[task_id] != EventStatus::EVENT_SUCCESS) {
        next_tasks.insert(task_id);
      }
    }
    if (FLAGS_caffe2_dag_net_collect_stats) {
      CAFFE_EVENT(
          stats_[CPU], poll_status_update_time_us, timer.MicroSeconds());
    }

    std::unordered_set<int> visited_children;
    for (auto& task_id : updated_tasks) {
      CAFFE_ENFORCE(
          status_[task_id] == EventStatus::EVENT_SCHEDULED ||
          status_[task_id] == EventStatus::EVENT_SUCCESS);

      for (auto& child_id : children(task_id)) {
        if (!visited_children.count(child_id)) {
          visited_children.insert(child_id);
          // Important - check whether we have already scheduled the task,
          // e.g. a child CUDA task can be scheduled after parent CUDA
          // task becomes EventStatus::EVENT_SCHEDULED and also later when
          // parent CUDA task becomes EventStatus::EVENT_SUCCESS
          if (!scheduled_tasks.count(child_id) && canSchedule(child_id)) {
            next_tasks.insert(child_id);
            scheduled_tasks.insert(child_id);
            schedule(child_id);
          }
        }
      }
    }

    current_tasks.swap(next_tasks);
  }
}

void AsyncPollingNet::updateTaskStats(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  if (status_[task_id] == EventStatus::EVENT_SCHEDULED) {
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_time_to_scheduled_us,
        task_timers_[task_id]->MicroSeconds());
  }
  if (status_[task_id] == EventStatus::EVENT_SUCCESS) {
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_time_to_succeeded_ms,
        task_timers_[task_id]->MilliSeconds());
  }
}

// Task specific implementations

int AsyncPollingNet::tasksNum() const {
  return chains_.size();
}

const Event& AsyncPollingNet::event(int task_id) const {
  auto& task = chains_[task_id];
  auto& last_task_op = operators_[task.back()];
  return last_task_op->event();
}

EventStatus AsyncPollingNet::query(int task_id) const {
  if (FLAGS_caffe2_dag_net_collect_stats) {
    Timer timer;
    auto status = event(task_id).Query();
    const auto& device_option = event(task_id).GetDeviceOption();
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_query_time_us,
        timer.MicroSeconds());
    return status;
  } else {
    return event(task_id).Query();
  }
}

bool AsyncPollingNet::isStreamFree(int task_id, int stream_id) const {
  auto& task = chains_[task_id];
  auto& last_task_op = operators_[task.back()];
  return last_task_op->IsStreamFree(stream_id);
}

const std::vector<int>& AsyncPollingNet::children(int task_id) const {
  const auto& task_node = chain_nodes_[task_id];
  return task_node.children_;
}

const std::vector<int>& AsyncPollingNet::parents(int task_id) const {
  const auto& task_node = chain_nodes_[task_id];
  return task_node.parents_;
}

bool AsyncPollingNet::canRunDependency(int parent_task_id, int child_task_id) {
  auto first_child_op_id = chains_[child_task_id].front();
  auto last_parent_op_id = chains_[parent_task_id].back();
  return operators_[last_parent_op_id]->event().CanSchedule(
      operators_[first_child_op_id]->event(),
      operators_[first_child_op_id]->SupportsAsyncScheduling());
}

void AsyncPollingNet::asyncWait(
    int task_id,
    int stream_id,
    const std::vector<int>& wait_task_ids) const {
  auto first_op_id = chains_[task_id].front();
  auto& first_op = operators_[first_op_id];
  std::vector<const Event*> events;
  events.reserve(wait_task_ids.size());
  for (auto wait_task_id : wait_task_ids) {
    events.push_back(&event(wait_task_id));
  }
  first_op->WaitEvents(events, stream_id);
}

void AsyncPollingNet::run(int task_id, int stream_id) {
  bool failed = false;
  std::string err_msg;
  for (auto& op_id : chains_[task_id]) {
    auto& op = operators_[op_id];
    try {
      bool result;
      if (FLAGS_caffe2_dag_net_collect_stats) {
        Timer timer;
        result = op->RunAsync(stream_id);
        CAFFE_EVENT(
            stats_[op->event().GetDeviceOption().device_type()],
            op_run_async_time_us,
            timer.MicroSeconds());
      } else {
        result = op->RunAsync(stream_id);
      }

      if (!result) {
        failed = true;
        err_msg = "Failed to execute task: op " +
            (op->has_debug_def() ? op->type() : " unknown");
        break;
      }
    } catch (const std::exception& e) {
      failed = true;
      err_msg = e.what();
      break;
    } catch (...) {
      failed = true;
      err_msg = "Failed to execute task: unknown error";
      break;
    }
  }

  if (failed) {
    has_chain_failed_ = true;
  }
}

AsyncPollingNet::~AsyncPollingNet() {}

CAFFE_DEFINE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPool,
    const DeviceOption&);

namespace {
std::shared_ptr<TaskThreadPool> AsyncNetCPUThreadPoolCreator(
    const DeviceOption& device_option) {
  CAFFE_ENFORCE_EQ(
      device_option.device_type(),
      CPU,
      "Unexpected device type for CPU thread pool");
  return GetAsyncNetCPUThreadPool();
}
} // namespace

CAFFE_REGISTER_CREATOR(ThreadPoolRegistry, CPU, AsyncNetCPUThreadPoolCreator);

/* static */
std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool() {
  static std::weak_ptr<TaskThreadPool> pool;
  static std::mutex pool_mutex;
  std::lock_guard<std::mutex> lock(pool_mutex);

  auto shared_pool = pool.lock();
  if (!shared_pool) {
    auto num_cores = std::thread::hardware_concurrency();
    CAFFE_ENFORCE(num_cores > 0, "Failed to get number of CPU cores");
    shared_pool = std::make_shared<TaskThreadPool>(num_cores);
    pool = shared_pool;
  }
  return shared_pool;
}

REGISTER_NET(async_polling, AsyncPollingNet);

} // namespace caffe2
