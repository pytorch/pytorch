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

#ifndef CAFFE2_CORE_NET_ASYNC_BASE_H_
#define CAFFE2_CORE_NET_ASYNC_BASE_H_

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

class AsyncNetBase : public NetBase {
 public:
  AsyncNetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~AsyncNetBase() override;

  bool SupportsAsync() override {
    return true;
  }

  vector<OperatorBase*> GetOperators() const override {
    return operators_;
  }

 protected:
  bool canSchedule(
      int chain_id,
      const std::vector<EventStatus>* status = nullptr);

  int tasksNum() const;
  Event& event(int task_id) const;
  EventStatus query(int task_id) const;
  const std::vector<int>& children(int task_id) const;
  const std::vector<int>& parents(int task_id) const;
  void asyncWait(
      int task_id,
      int stream_id,
      const std::vector<int>& wait_task_ids) const;
  bool run(int task_id, int stream_id);
  int stream(int task_id);
  std::shared_ptr<TaskThreadPool> pool(const DeviceOption& device_option);

  void finishTasks(const std::unordered_set<int>& task_ids);
  void finalizeEvents();

  bool isStreamFree(int task_id, int stream_id) const;

  // Operator/task graph
  std::vector<OperatorBase*> operators_;
  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<std::vector<int>> chains_;
  std::vector<dag_utils::OpGraphNode> chain_nodes_; // chains' parents/children

  // Pools and streams
  std::mutex pools_mutex_;
  std::vector<std::shared_ptr<TaskThreadPool>> gpu_pools_;
  std::shared_ptr<TaskThreadPool> cpu_pool_;
  std::shared_ptr<TaskThreadPool> gpu_pool_;
  static thread_local std::vector<int> stream_counters_;

  DISABLE_COPY_AND_ASSIGN(AsyncNetBase);
};

CAFFE_DECLARE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPool,
    const DeviceOption&);

std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool();

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_POLLING_H_
