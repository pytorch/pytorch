#ifndef CAFFE2_CORE_NET_ASYNC_BASE_H_
#define CAFFE2_CORE_NET_ASYNC_BASE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/thread_pool.h"

namespace caffe2 {

class AsyncNetExecutorHelper;

namespace tracing {
class Tracer;
}

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

  bool RunAsync() override;

 protected:
  bool canSchedule(
      int chain_id,
      const std::vector<EventStatus>* status = nullptr);

  int tasksNum() const;
  Event& event(int task_id) const;
  EventStatus query(int task_id) const;
  const std::vector<int>& children(int task_id) const;
  const std::vector<int>& parents(int task_id) const;
  int num_ops(int task_id) const;
  void asyncWait(
      int task_id,
      int stream_id,
      const std::vector<int>& wait_task_ids) const;
  void run(int task_id, int stream_id);
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
  // first int key - device id, second - pool size, one pool per (device, size)
  typedef std::unordered_map<
      int,
      std::unordered_map<int, std::shared_ptr<TaskThreadPool>>>
      PoolsMap;
  PoolsMap cpu_pools_;
  PoolsMap gpu_pools_;
  static thread_local std::vector<int> stream_counters_;
  int num_workers_;

  // Tracing
  void initTracer(const std::shared_ptr<const NetDef>& net_def);
  Timer timer_;
  bool trace_net_;
  bool trace_batch_;
  int batch_iter_;
  std::unique_ptr<tracing::Tracer> tracer_;

  DISABLE_COPY_AND_ASSIGN(AsyncNetBase);

 private:
  OperatorBase* op(int op_idx) const;

  std::shared_ptr<TaskThreadPool>
  pool_getter(PoolsMap& pools, int device_type, int device_id, int pool_size);

  std::unique_ptr<AsyncNetExecutorHelper> helper_;

  friend class AsyncNetExecutorHelper;
  friend class tracing::Tracer;
};

CAFFE_DECLARE_SHARED_REGISTRY(ThreadPoolRegistry, TaskThreadPool, int, int);

class AsyncNetExecutorHelper : public ExecutorHelper {
 public:
  explicit AsyncNetExecutorHelper(AsyncNetBase* net) : net_(net) {}
  std::shared_ptr<TaskThreadPool> GetPool(
      const DeviceOption& option) const override {
    return net_->pool(option);
  }

 private:
  AsyncNetBase* net_;
};

std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool(
    int numa_node_id,
    int pool_size);

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_POLLING_H_
