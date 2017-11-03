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

#ifndef CAFFE2_CORE_NET_ASYNC_POLLING_H_
#define CAFFE2_CORE_NET_ASYNC_POLLING_H_

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/thread_pool.h"

namespace caffe2 {

class AsyncPollingNet : public NetBase {
 public:
  AsyncPollingNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~AsyncPollingNet() override;

  bool SupportsAsync() override {
    return true;
  }

  vector<OperatorBase*> GetOperators() const override {
    return operators_;
  }

  void Wait() override;

 protected:
  bool DoRunAsync() override;

 private:
  void pollAndSchedule();
  bool canSchedule(int chain_id);
  void schedule(int task_id);

  int tasksNum() const;
  const Event& event(int task_id) const;
  EventStatus query(int task_id) const;
  const std::vector<int>& children(int task_id) const;
  const std::vector<int>& parents(int task_id) const;
  void asyncWait(
      int task_id,
      int stream_id,
      const std::vector<int>& wait_task_ids) const;
  void run(int task_id, int stream_id);
  int stream(int task_id);
  std::shared_ptr<TaskThreadPool> pool(const DeviceOption& device_option);
  bool canRunDependency(int parent_task_id, int child_task_id);

  // Operator/task graph
  std::vector<OperatorBase*> operators_;
  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<std::vector<int>> chains_;
  std::vector<dag_utils::OpGraphNode> chain_nodes_; // chains' parents/children

  // Synchronization
  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;

  // Pools and streams
  std::unordered_map<DeviceOption, std::shared_ptr<TaskThreadPool>> pools_;
  std::vector<int> stream_rr_counters_;

  // Stats
  struct AsyncPollingNetStats {
    CAFFE_STAT_CTOR(AsyncPollingNetStats);
    CAFFE_AVG_EXPORTED_STAT(poll_time_ms);
    CAFFE_AVG_EXPORTED_STAT(task_pool_wait_time_us);
    CAFFE_AVG_EXPORTED_STAT(op_run_async_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_query_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_run_time_us);
    CAFFE_AVG_EXPORTED_STAT(poll_status_update_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_scheduled_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_succeeded_ms);
  };
  mutable std::vector<AsyncPollingNetStats> stats_;
  std::vector<std::unique_ptr<Timer>> task_timers_;
  void updateTaskStats(int task_id);

  // Polling
  std::vector<EventStatus> status_;
  std::atomic<bool> has_chain_failed_;
  void reset();

  DISABLE_COPY_AND_ASSIGN(AsyncPollingNet);
};

CAFFE_DECLARE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPool,
    const DeviceOption&);

std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool();

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_POLLING_H_
