#ifndef CAFFE2_CORE_NET_ASYNC_BASE_H_
#define CAFFE2_CORE_NET_ASYNC_BASE_H_

#include "c10/core/thread_pool.h"
#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/prof_dag_counters.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/prof_dag.pb.h"
#include "caffe2/utils/proto_utils.h"

C10_DECLARE_int(caffe2_streams_per_gpu);
C10_DECLARE_int(caffe2_net_async_max_gpus);
C10_DECLARE_int(caffe2_net_async_max_numa_nodes);
C10_DECLARE_int(caffe2_net_async_thread_pool_size);
C10_DECLARE_bool(caffe2_net_async_check_stream_status);
C10_DECLARE_bool(caffe2_net_async_use_single_pool);
C10_DECLARE_bool(caffe2_net_async_use_per_net_pools);
C10_DECLARE_bool(caffe2_net_async_run_root_tasks_inline);
C10_DECLARE_bool(caffe2_net_async_profile_operators);

namespace caffe2 {

class AsyncNetExecutorHelper;

namespace tracing {
class Tracer;
}

struct ExecutionOptions {
  explicit ExecutionOptions(const std::shared_ptr<const NetDef>& net_def);

  // number of gpu streams per gpu per cpu thread
  int streams_per_gpu_ = 1;
  // ops synchronization options
  bool finish_chain_ = false;
  bool always_schedule_child_ = false;
  // try to pick gpu stream that is not busy
  bool check_stream_status_ = false;
  // use single thread pool for all devices
  bool use_single_pool_ = false;
  // use per net instances thread pools instead of global ones
  bool use_per_net_pools_ = false;
  // whether RunAsync is blocking
  bool is_blocking_ = false;
  // prof_dag counters reporting
  bool report_stats_ = false;
  // immediately run children tasks inline whenever possible
  bool use_dfs_scheduling_ = false;
  // run net's root tasks in RunAsync thread instead of in thread pool
  bool run_root_tasks_inline_ = false;
};

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

  ProfDAGProtos GetOperatorStats() const;
  ProfDAGProtos GetPerOperatorCost() const;
  ProfDAGReport GetProfReport() const;

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

  int firstTaskOpId(int task_id) const;
  int lastTaskOpId(int task_id) const;
  const OperatorBase* firstTaskOp(int task_id) const;
  const OperatorBase* lastTaskOp(int task_id) const;
  OperatorBase* firstTaskOp(int task_id);
  OperatorBase* lastTaskOp(int task_id);

  void asyncWait(
      int task_id,
      int stream_id,
      const std::vector<int>& wait_task_ids) const;
  bool run(int task_id, int stream_id) noexcept;
  int stream(int task_id);
  TaskThreadPoolBase* pool(const DeviceOption& device_option);
  TaskThreadPoolBase* pool();

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
      std::unordered_map<int, std::shared_ptr<TaskThreadPoolBase>>>
      PoolsMap;
  PoolsMap cpu_pools_;
  PoolsMap gpu_pools_;
  static std::vector<int>& getStreamCounters();
  int num_workers_;

  // Exception/error handling
  void handleChainError(
      int task_id,
      OperatorBase* op,
      const char* err_msg,
      bool save_exception = false) noexcept;
  std::atomic<bool> success_;

  // Tracing
  std::shared_ptr<tracing::Tracer> tracer_;

  // execution mode flags
  ExecutionOptions options_;

  ProfDAGCounters counters_;

  C10_DISABLE_COPY_AND_ASSIGN(AsyncNetBase);

 private:
  TaskThreadPoolBase*
  poolGetter(PoolsMap& pools, int device_type, int device_id, int pool_size);

  std::unique_ptr<AsyncNetExecutorHelper> helper_;

  friend class AsyncNetExecutorHelper;
  friend class tracing::Tracer;
};

class AsyncNetExecutorHelper : public ExecutorHelper {
 public:
  explicit AsyncNetExecutorHelper(AsyncNetBase* net) : net_(net) {}
  TaskThreadPoolBase* GetPool(const DeviceOption& option) const override {
    return net_->pool(option);
  }

 private:
  AsyncNetBase* net_;
};

template <class TaskThreadPoolImpl, int device_type>
std::shared_ptr<TaskThreadPoolBase>
GetAsyncNetThreadPool(int device_id, int pool_size, bool create_new) {
  static std::unordered_map<
      int,
      std::unordered_map<int, std::weak_ptr<TaskThreadPoolBase>>>
      pools;
  static std::mutex pool_mutex;

  const auto& device_type_name = DeviceTypeName(device_type);

  if (pool_size <= 0) {
    if (FLAGS_caffe2_net_async_thread_pool_size > 0) {
      pool_size = FLAGS_caffe2_net_async_thread_pool_size;
      LOG(INFO) << "Using default " << device_type_name
                << " pool size: " << pool_size << "; device id: " << device_id;
    } else {
      auto num_cores = std::thread::hardware_concurrency();
      CAFFE_ENFORCE(num_cores > 0, "Failed to get number of CPU cores");
      LOG(INFO) << "Using estimated " << device_type_name
                << " pool size: " << num_cores << "; device id: " << device_id;
      pool_size = num_cores;
    }
  } else {
    LOG(INFO) << "Using specified " << device_type_name
              << " pool size: " << pool_size << "; device id: " << device_id;
  }

  if (create_new) {
    LOG(INFO) << "Created new " << device_type_name
              << " pool, size: " << pool_size << "; device id: " << device_id;
    return std::make_shared<TaskThreadPoolImpl>(pool_size, device_id);
  } else {
    std::lock_guard<std::mutex> lock(pool_mutex);

    auto shared_pool = pools[device_id][pool_size].lock();
    if (!shared_pool) {
      LOG(INFO) << "Created shared " << device_type_name
                << " pool, size: " << pool_size << "; device id: " << device_id;
      shared_pool = std::make_shared<TaskThreadPoolImpl>(pool_size, device_id);
      pools[device_id][pool_size] = shared_pool;
    }
    return shared_pool;
  }
}

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_BASE_H_
