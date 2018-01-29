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

#ifndef CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_
#define CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_

#include "caffe2/core/net_async_base.h"

namespace caffe2 {

class AsyncSchedulingNet : public AsyncNetBase {
 public:
  AsyncSchedulingNet(
      const std::shared_ptr<const NetDef>& net_def,
      Workspace* ws);
  ~AsyncSchedulingNet() override;

  void Wait() override;

 protected:
  bool DoRunAsync() override;

  void pollAndSchedule(int task_id);
  void schedule(int task_id);
  void reset();
  void finishRun();
  int updateParentCount(int child_id);

  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;
  std::atomic<bool> success_;

  std::mutex cleanup_mutex_;
  std::atomic<bool> cleanup_;

  std::atomic<int> processed_tasks_num_;

  DISABLE_COPY_AND_ASSIGN(AsyncSchedulingNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_
