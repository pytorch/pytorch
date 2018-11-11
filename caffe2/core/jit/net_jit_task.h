#ifndef CAFFE2_NET_JIT_TASK_H
#define CAFFE2_NET_JIT_TASK_H

#include "caffe2/core/event.h"
#include "caffe2/core/jit/net_jit.h"
#include "caffe2/core/jit/net_jit_future.h"
#include "caffe2/core/jit/net_jit_task_runner.h"

namespace caffe2 {

class JITC2Task {
 public:
  JITC2Task(
      const std::shared_ptr<JITC2Program>& program,
      const size_t start_address,
      JITC2TaskRunner& task_runner,
      bool use_dfs_scheduling = false,
      const std::vector<JITFuture*>& inputs = {},
      const std::unordered_map<int, JITFuture*>& local_vars = {});

  JITFuture* Run();

  JITFuture& GetFuture();
  const JITFuture& GetFuture() const;

  const DeviceOption& GetDeviceOption() const;

 private:
  bool handleC2Op(const JITOp& op);
  bool handleFork(const JITOp& op);
  bool handleJoin(const JITOp& op);
  bool handleReturn(const JITOp& op);
  void init();
  std::vector<JITFuture*> futures(const std::vector<int>& future_ids);
  bool handleOp(const JITOp& op);

  std::shared_ptr<JITC2Program> program_;
  size_t start_address_;
  size_t address_;
  std::vector<JITFuture*> inputs_;
  JITC2TaskRunner& task_runner_;
  class DeviceOption device_option_;
  JITFuture future_;
  std::unique_ptr<JITFuture> joined_future_;
  Event* cpu_async_event_;

  // holds "local variables", in C2 case - generated futures
  std::unordered_map<int, JITFuture*> local_vars_;
  bool use_dfs_scheduling_;
  std::vector<std::shared_ptr<JITC2Task>> inline_tasks_;

  std::unordered_map<int, std::function<bool(const JITOp&)>> handler_registry_;
};

} // namespace caffe2

#define _REGISTER_OP_HANDLER(op, func) \
  handler_registry_[op] =              \
      std::bind(&JITC2Task::func, this, std::placeholders::_1);

#endif
