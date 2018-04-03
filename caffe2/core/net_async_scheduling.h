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
  virtual void finishRun();
  int updateParentCount(int child_id);

  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;
  std::atomic<bool> success_;

  std::mutex cleanup_mutex_;
  std::atomic<bool> cleanup_;

  std::atomic<int> processed_tasks_num_;
  std::mutex exception_mutex_;
  std::vector<std::string> exception_messages_;

  DISABLE_COPY_AND_ASSIGN(AsyncSchedulingNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_
