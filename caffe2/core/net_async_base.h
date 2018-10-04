#ifndef CAFFE2_CORE_NET_ASYNC_BASE_H_
#define CAFFE2_CORE_NET_ASYNC_BASE_H_

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/thread_pool.h"

C10_DECLARE_int(caffe2_streams_per_gpu);
C10_DECLARE_bool(caffe2_net_async_finish_chain);
C10_DECLARE_bool(caffe2_net_async_always_schedule_child);
C10_DECLARE_int(caffe2_net_async_max_gpus);
C10_DECLARE_int(caffe2_net_async_max_numa_nodes);
C10_DECLARE_int(caffe2_net_async_cpu_pool_size);
C10_DECLARE_bool(caffe2_net_async_check_stream_status);
C10_DECLARE_bool(caffe2_net_async_use_single_pool);
C10_DECLARE_bool(caffe2_net_async_use_per_net_pools);

namespace caffe2 {

class AsyncNetExecutorHelper;

namespace tracing {
class Tracer;
}

class CAFFE2_API AsyncNetBase : public NetBase {
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

  const dag_utils::ExecutionChains& TEST_execution_chains() const {
    return execution_chains_;
  }

 protected:
  bool canSchedule(
      int chain_id,
      const std::vector<EventStatus>* status = nullptr,
      bool* parent_failed = nullptr);
  bool canSchedule(int parent_id, int child_id);

  int tasksNum() const;
  Event& event(int task_id) const;
  EventStatus query(int task_id) const;
  const std::vector<int>& children(int task_id) const;
  const std::vector<int>& parents(int task_id) const;
  int updateParentCount(int child_id);
  int getParentCount(int child_id);
  bool testAndSetScheduled(int task_id);
  int numOps(int task_id) const;
  const OperatorBase* firstTaskOp(int task_id) const;
  const OperatorBase* lastTaskOp(int task_id) const;

  void asyncWait(
      int task_id,
      int stream_id,
      const std::vector<int>& wait_task_ids) const;
  bool run(int task_id, int stream_id);
  int stream(int task_id);
  TaskThreadPool* pool(const DeviceOption& device_option);

  void finishTasks(const std::unordered_set<int>& task_ids);
  void finalizeEvents();

  bool isStreamFree(int task_id, int stream_id) const;

  virtual void reset();

  bool handleRunError() override;

  // Operator/task graph
  std::vector<OperatorBase*> operators_;
  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<std::vector<int>> chains_;
  std::vector<dag_utils::OpGraphNode> chain_nodes_; // chains' parents/children
  dag_utils::ExecutionChains execution_chains_; // for testing

  // Pools and streams
  std::mutex pools_mutex_;
  // first int key - device id, second - pool size, one pool per (device, size)
  typedef std::unordered_map<
      int,
      std::unordered_map<int, std::shared_ptr<TaskThreadPool>>>
      PoolsMap;
  PoolsMap cpu_pools_;
  PoolsMap gpu_pools_;
  static std::vector<int>& getStreamCounters();
  int num_workers_;

  // Exception/error handling
  void setTaskErrorMessage(int task_id, const std::string& err_msg);
  std::atomic<bool> success_;
#ifdef CAFFE2_USE_EXCEPTION_PTR
  // Mutex that protects caught_exception_
  std::mutex exception_mutex_;
  std::exception_ptr caught_exception_;
#endif // CAFFE2_USE_EXCEPTION_PTR

  // Tracing
  std::shared_ptr<tracing::Tracer> tracer_;

  // execution mode flags
  void computeExecutionModeFlags();
  int streams_per_gpu_;
  bool finish_chain_;
  bool always_schedule_child_;
  bool check_stream_status_;
  bool use_single_pool_;
  bool use_per_net_pools_;
  bool is_blocking_;

  C10_DISABLE_COPY_AND_ASSIGN(AsyncNetBase);

 private:
  void storeExceptionPtr();

  TaskThreadPool*
  poolGetter(PoolsMap& pools, int device_type, int device_id, int pool_size);

  std::unique_ptr<AsyncNetExecutorHelper> helper_;

  friend class AsyncNetExecutorHelper;
  friend class tracing::Tracer;
};

C10_DECLARE_SHARED_REGISTRY(ThreadPoolRegistry, TaskThreadPool, int, int, bool);

class AsyncNetExecutorHelper : public ExecutorHelper {
 public:
  explicit AsyncNetExecutorHelper(AsyncNetBase* net) : net_(net) {}
  TaskThreadPool* GetPool(const DeviceOption& option) const override {
    return net_->pool(option);
  }

 private:
  AsyncNetBase* net_;
};

std::shared_ptr<TaskThreadPool>
GetAsyncNetCPUThreadPool(int numa_node_id, int pool_size, bool create_new);

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_BASE_H_
