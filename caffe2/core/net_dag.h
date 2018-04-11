#ifndef CAFFE2_CORE_NET_DAG_H_
#define CAFFE2_CORE_NET_DAG_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

class DAGNetBase : public NetBase {
 public:
  DAGNetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~DAGNetBase() override;

  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  const dag_utils::ExecutionChains& TEST_execution_chains() const {
    return execution_chains_;
  }

  vector<OperatorBase*> GetOperators() const override {
    return operators_;
  }

 protected:
  bool DoRunAsync() override;

  virtual bool RunAt(int chain_id, const std::vector<int>& chain) = 0;
  void HandleException(int operator_idx, const std::string& exception_str);

  vector<dag_utils::OperatorNode> operator_nodes_;
  vector<OperatorBase*> operators_;
  dag_utils::ExecutionChains execution_chains_;
  vector<int> initial_frontier_;
  std::unique_ptr<SimpleQueue<int>> job_queue_;
  std::vector<std::thread> workers_;
  int num_workers_;
  int remaining_ops_;

  bool success_;
  // Use an atomic to guard caught_exception_ so it is written to only once
  std::atomic<bool> caught_exception_yet_;
#ifdef CAFFE2_USE_EXCEPTION_PTR
  std::exception_ptr caught_exception_;
#endif // CAFFE2_USE_EXCEPTION_PTR
  int iter_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;
  std::mutex run_in_progress_;

  struct DAGNetStats {
    CAFFE_STAT_CTOR(DAGNetStats);
    CAFFE_AVG_EXPORTED_STAT(task_pool_wait_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_scheduled_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_succeeded_ms);
    CAFFE_AVG_EXPORTED_STAT(task_wait_time_us);
  };
  mutable std::vector<DAGNetStats> stats_;
  std::unordered_map<int, std::unique_ptr<Timer>> task_timers_;

  DISABLE_COPY_AND_ASSIGN(DAGNetBase);
};

class DAGNet : public DAGNetBase {
 public:
  using DAGNetBase::DAGNetBase;

 protected:
  bool RunAt(int chain_id, const std::vector<int>& chain) override;
  bool SupportsAsync() override {
    return false;
  }
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_DAG_H_
