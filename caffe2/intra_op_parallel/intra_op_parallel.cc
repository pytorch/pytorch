#include "intra_op_parallel.h"

#include <functional>

// #define INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif
#ifdef CAFFE2_INTRA_OP_PARALLEL_CAN_USE_TBB
#include <tbb/parallel_for.h>
#endif

C10_DEFINE_int(
    caffe2_intra_op_parallel_max_num_tasks,
    65536,
    "The maximum number of tasks used for intra-op parallelism per operator");
C10_DEFINE_int(
    caffe2_intra_op_parallel_max_num_workers,
    65536,
    "The maximum number of tasks used for intra-op parallelism per operator. "
    "Deprecated: please use caffe2_intra_op_parallel_max_num_tasks instead");
C10_DEFINE_bool(
    caffe2_intra_op_parallel_only_grab_idle_threads,
    true,
    "When true, only grab idle threads to minimize over-subscription");

using namespace std;

namespace caffe2 {

namespace intra_op_parallel {

ParallelOpBase::ParallelOpBase(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws), count_(0) {
#ifdef CAFFE2_INTRA_OP_PARALLEL_CAN_USE_TBB
  use_tbb_ = operator_def.engine() == "TBB";
#endif

  max_num_tasks_ = std::max(
      this->GetSingleArgument<int>("max_num_tasks", -1),
      this->GetSingleArgument<int>("max_num_workers", -1));
  if (max_num_tasks_ == -1) {
    max_num_tasks_ = std::min(
        FLAGS_caffe2_intra_op_parallel_max_num_tasks,
        FLAGS_caffe2_intra_op_parallel_max_num_workers);
  } else {
    max_num_tasks_ = std::min(
        max_num_tasks_,
        std::min(
            FLAGS_caffe2_intra_op_parallel_max_num_tasks,
            FLAGS_caffe2_intra_op_parallel_max_num_workers));
  }
}

ParallelOpBase::~ParallelOpBase() {}

mutex& ParallelOpBase::TaskPoolMutex(int numa_node_id) {
  // additional 1 for numa_node_id == -1
  static vector<mutex> mutexes(std::max(0, GetNumNUMANodes()) + 1);
  CAFFE_ENFORCE_GE(numa_node_id, -1);
  CAFFE_ENFORCE_LT(numa_node_id, std::max(GetNumNUMANodes(), 0));
  return mutexes[numa_node_id + 1];
}

bool ParallelOpBase::RunOnDevice() {
#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  t_begin_ = chrono::system_clock::now();
#endif

  int num_tasks = 1;
  ExecutorHelper* executor_helper = OperatorBase::GetExecutorHelper();
  if (executor_helper) {
    TaskThreadPoolBase* pool = executor_helper->GetPool(device_option());
    assert(pool);

    // In TBB we can't know the number of idle worker threads
    num_tasks = std::min<int>(
        max_num_tasks_,
        FLAGS_caffe2_intra_op_parallel_only_grab_idle_threads && !use_tbb_
            ? pool->numAvailable() + 1
            : pool->size());
  }

  // Make sure RunOnDevicePrologue has allocated all outputs with mutable_data.
  // This is because resizing of tensor that can be trigerred by mutable_data
  // is not thread safe.
  if (!RunOnDevicePrologue(num_tasks)) {
    return false;
  }

  // Make sure RunOnDevicePrologue has allocated all outputs with mutable_data.
  // This is because resizing of tensor that can be trigerred by mutable_data
  // is not thread safe.
  for (int i = 0; i < OutputSize(); ++i) {
    Output(i)->raw_mutable_data();
  }

  tracing::TracerGuard* curr_tracer_guard =
      tracing::TracerGuard::getCurrentTracerGuard();
  if (curr_tracer_guard) {
    tracer_guard_ = *curr_tracer_guard;
  } else {
    // The default non-enabled tracer guard
    tracer_guard_ = tracing::TracerGuard();
  }

#ifdef CAFFE2_INTRA_OP_PARALLEL_CAN_USE_TBB
  if (use_tbb_) {
    ::tbb::parallel_for(
        0,
        num_tasks,
        [this, num_tasks](size_t task_id) {
          RunOnDeviceParallelWrapper_(task_id, num_tasks);
        },
        ::tbb::simple_partitioner());
    return true;
  } else
#endif
  {
    if (executor_helper) {
      TaskThreadPoolBase* pool = executor_helper->GetPool(device_option());
      assert(pool);

      lock_guard<mutex> lock(TaskPoolMutex(
          device_option().has_numa_node_id() ? device_option().numa_node_id()
                                             : -1));
      for (int task_id = 1; task_id < num_tasks; ++task_id) {
        pool->run(bind(
            &ParallelOpBase::RunOnDeviceParallelWrapper_,
            this,
            task_id,
            num_tasks));
      }
    }

    return RunOnDeviceParallelWrapper_(0, num_tasks);
  }
}

bool ParallelOpBase::RunOnDeviceParallelWrapper_(int task_id, int num_tasks) {
  if (0 == task_id) {
    VLOG(2) << "Executing " << debug_def().type() << " with " << num_tasks
            << " tasks";
  }

#ifdef CAFFE2_USE_MKL
  // Make sure mkl_threads is 1 to avoid spawning too many threads with nested
  // parallelization (mkl_threads overrides omp_threads).
  int old_mkl_num_threads = mkl_set_num_threads_local(1);
#endif

  bool ret;

  if (task_id > 0) {
    tracing::TracerGuard task_tracer_guard = tracer_guard_;
    task_tracer_guard.recordEventStart();
    ret = RunOnDeviceParallel(task_id, num_tasks);
  } else {
    // task 0 is inlined so we don't need to do create another TracerGuard
    ret = RunOnDeviceParallel(task_id, num_tasks);
  }

#ifdef CAFFE2_USE_MKL
  mkl_set_num_threads_local(old_mkl_num_threads);
#endif

  if (++count_ == num_tasks) {
    count_ = 0;
    if (!use_tbb_) {
      // Without TBB, we need to use C2 async execution.
      event().SetFinished();
    }
    // Need to explicitly disable tracer guard.
    // Otherwise, segfault at TracerGuard destructor.
    tracer_guard_.disable();

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
    chrono::time_point<chrono::system_clock> t_end =
        chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin_).count();
    if (debug_def().input().size() > 1) {
      LOG(INFO) << debug_def().type() << " with input blob "
                << debug_def().input(0) << " takes " << dt << " sec";
    } else {
      LOG(INFO) << debug_def().type() << " takes " << dt << " sec";
    }
#endif
  }
  return ret;
}

} // namespace intra_op_parallel

} // namespace caffe2
