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

#include "caffe2/core/net_async_base.h"

namespace caffe2 {

class AsyncPollingNet : public AsyncNetBase {
 public:
  AsyncPollingNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~AsyncPollingNet() override;

 protected:
  bool DoRunAsync() override;

  bool pollAndSchedule();
  void schedule(int task_id);

  // Synchronization
  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;

  // Stats
  struct AsyncPollingNetStats {
    CAFFE_STAT_CTOR(AsyncPollingNetStats);
    CAFFE_AVG_EXPORTED_STAT(poll_time_ms);
    CAFFE_AVG_EXPORTED_STAT(task_pool_wait_time_us);
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
  void reset();
  std::atomic<bool> has_chain_failed_;

  DISABLE_COPY_AND_ASSIGN(AsyncPollingNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_POLLING_H_
