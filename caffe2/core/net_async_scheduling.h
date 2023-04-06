#ifndef CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_
#define CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_

#include "caffe2/core/net_async_base.h"

namespace caffe2 {

class TORCH_API AsyncSchedulingNet : public AsyncNetBase {
 public:
  AsyncSchedulingNet(
      const std::shared_ptr<const NetDef>& net_def,
      Workspace* ws);
  ~AsyncSchedulingNet() override;

  void Wait() override;

  void Cancel() override;

 protected:
  bool RunAsync() override;

  void pollAndSchedule(int task_id);
  void schedule(int task_id, bool run_inline = false) noexcept;
  void reset() override;
  virtual void finishRun();
  void parentCallback(int parent_id);
  bool isInlineTask(int parent_id, int child_id) const;

  void CancelAndFinishAsyncTasks();

  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;

  std::atomic<int> processed_tasks_num_;

  C10_DISABLE_COPY_AND_ASSIGN(AsyncSchedulingNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_SCHEDULING_H_
