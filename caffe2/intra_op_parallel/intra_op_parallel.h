#pragma once

#include <chrono>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace intra_op_parallel {

// TODO: set this in cmake later when moving to open source
#define INTRA_OP_PARALLEL_CAN_USE_TBB

/**
 * The base class of parallel Caffe2 operators.
 *
 * Instead of RunOnDevice, the derived classes should implement
 * RunOnDevicePrologue and RunOnDeviceParallel.
 *
 * RunOnDevicePrologue(int num_tasks) is invoked only once per operator
 * execution and everything that is not thread-safe (e.g., determining the
 * shape of output tensor) should be done here.
 *
 * RunOnDeviceParallel(int task_id, int num_tasks) is invoked by num_tasks
 * times with different task_id ranging from 0 to num_tasks - 1.
 * Whether RunOnDeviceParallel with smaller task_id will be invoked before
 * the one with bigger task_id is up to the Caffe executor and a parallel
 * Caffe2 operator implementation should never rely on such assumption.
 * It's up to the implementation of derived class how to partition the
 * computation using task_id and num_tasks.
 * A simple example of partitioning can be found from
 * intra_op_parallel_lengths_reducer_ops.h .
 * partition.h provides helper functions for 1D and 2D decompositions.
 *
 * num_tasks, the degree of parallelism, is determined by the followings:
 * num_tasks <= max_num_workers argument of the operator
 * num_tasks <= FLAGS_caffe2_intra_op_parallel_max_num_workers
 * when caffe2_intra_op_parallel_only_grab_idle_threads == true (default),
 *   num_tasks <= (the number of idle threads in the pool obtained by
 *                TaskThreadPoolBase::numAvailable()) + 1
 *   We add +1 here for the "main" thread already grabbed for the operator
 * otherwise, num_tasks <= the number of all threads in the pool obtained by
 *                         TaskThreadPoolBase::size()
 *
 * A parallel Caffe2 operator is async (i.e. HasSyncPart returns true) and
 * this is in order to signal the completion of all the tasks by simply having
 * the task invoked the last time call event().SetFinished() instead of each
 * task is actively waiting for all the other tasks (for example using a
 * barrier synchronization).
 * The latter approach would be deadlock prone.
 */
class ParallelOpBase : public Operator<CPUContext> {
 public:
  ParallelOpBase(const OperatorDef& operator_def, Workspace* ws);
  virtual ~ParallelOpBase() {}

  /**
   * A mutex to synchronize the thread pool for numa_node_id
   *
   * When enqueuing multiple tasks to a thread pool that synchronize with
   * each other (e.g., via a barrier), locking this mutex avoids deadlock.
   * An example of deadlock when we don't lock: the thread pool has 2 threads.
   * A parallel op A enqueues 2 tasks A0 and A1, and another parallel op B
   * enqueues 2 tasks B0 and B1. If the order of enqueue happens to be A0->B0->
   * A1->B1, and A0<->A1 and B0<->B1 synchornize, A0 and B0 will be scheduled
   * while A0 indefinitely waiting for A1, and B0 indefinitely waiting for B1,
   * leading to a deadlock.
   *
   * We also recommend not having any synchronization among the tasks from
   * a single parallel operator and instead split the the task into multiple
   * pieces and rely on scheduling of the executor for synchronization (see
   * ParallelFullyConnectedGradientOp for an example).
   * Even though not having synchronization among the tasks from a single
   * parallel operator will avoid deadlocks, atomically enqueueing the tasks
   * (i.e. gang scheduling) has a performance benefit.
   *
   * When an operator needs to enqueue tasks to multiple thread pools across
   * multiple devices (e.g., NUMAAllReduceOp), we need to 1) lock the mutexes
   * in an increasing order, 2) enqueue tasks, and 3) unlock the mutexes (the
   * order of unlocking doesn't matter to avoid deadlock).
   */
  static std::mutex& TaskPoolMutex(int numa_node_id);

 protected:
  virtual bool RunOnDevicePrologue(int num_tasks) = 0;
  virtual bool RunOnDeviceParallel(int task_id, int num_tasks) = 0;

  int max_num_tasks_;
  bool use_tbb_{false}; // are we using TBB task graph?

 private:
  bool RunOnDevice() final;
  bool HasAsyncPart() const final {
    // With TBB, we don't need to rely on C2 async execution
    return !use_tbb_;
  }

  bool RunOnDeviceParallelWrapper_(int task_id, int num_tasks);

  std::chrono::time_point<std::chrono::system_clock> t_begin_;
  std::atomic<int> count_; // keep track of the number of finished tasks
};

template <typename Sizes, typename... ExtraArgs>
struct DispatchHelper;

template <int FirstVal, int... Values, typename... ExtraArgs>
struct DispatchHelper<FixedValues<FirstVal, Values...>, ExtraArgs...> {
  template <typename Op>
  static bool call(Op* op, int value, int task_id, int num_tasks) {
    if (FirstVal == value) {
      return op->template DoRunWithValue<ExtraArgs..., FirstVal>(
          task_id, num_tasks);
    }
    return DispatchHelper<FixedValues<Values...>, ExtraArgs...>::template call<
        Op>(op, value, task_id, num_tasks);
  }
};

template <typename... ExtraArgs>
struct DispatchHelper<FixedValues<>, ExtraArgs...> {
  template <typename Op>
  static bool call(Op* op, int64_t /*size*/, int task_id, int num_tasks) {
    return op->template DoRunWithValue<ExtraArgs..., -1>(task_id, num_tasks);
  }
};

#define C10_DEFINE_TENSOR_TYPES_DISPATCHER(                                    \
    TensorTypes, DoRunWithType, DoRunWithOtherType)                            \
  template <typename FirstType, typename... Types, typename... ExtraArgs>      \
  struct DispatchHelper<TensorTypes<FirstType, Types...>, ExtraArgs...> {      \
    template <typename Op>                                                     \
    static bool                                                                \
    call(Op* op, const TypeMeta& meta, int task_id, int num_tasks) {           \
      static_assert(                                                           \
          !std::is_same<GenericTensorImplementation, FirstType>::value,        \
          "GenericTensorImplementation must be the last in TensorTypes list"); \
      if (meta.Match<FirstType>()) {                                           \
        return op->template DoRunWithType<ExtraArgs..., FirstType>(            \
            task_id, num_tasks);                                               \
      }                                                                        \
      return DispatchHelper<TensorTypes<Types...>, ExtraArgs...>::             \
          template call<Op>(op, meta, task_id, num_tasks);                     \
    }                                                                          \
    template <typename Op>                                                     \
    static bool                                                                \
    call(Op* op, const Tensor& tensor, int task_id, int num_tasks) {           \
      return call<Op>(op, tensor.dtype(), task_id, num_tasks);                 \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob, int task_id, int num_tasks) {   \
      return call<Op>(op, blob.meta(), task_id, num_tasks);                    \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename... ExtraArgs>                                             \
  struct DispatchHelper<TensorTypes<>, ExtraArgs...> {                         \
    template <typename Op>                                                     \
    static bool                                                                \
    call(Op* /* unused */, const TypeMeta& meta, int task_id, int num_tasks) { \
      CAFFE_THROW(                                                             \
          "Unsupported type of tensor: ", meta.name(), task_id, num_tasks);    \
    }                                                                          \
    template <typename Op>                                                     \
    static bool                                                                \
    call(Op* op, const Tensor& tensor, int task_id, int num_tasks) {           \
      return call<Op>(op, tensor.dtype(), task_id, num_tasks);                 \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob, int task_id, int num_tasks) {   \
      return call<Op>(op, blob.meta(), task_id, num_tasks);                    \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename... ExtraArgs>                                             \
  struct DispatchHelper<                                                       \
      TensorTypes<GenericTensorImplementation>,                                \
      ExtraArgs...> {                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const TypeMeta&, int task_id, int num_tasks) {    \
      return op->template DoRunWithOtherType<ExtraArgs...>(                    \
          task_id, num_tasks);                                                 \
    }                                                                          \
    template <typename Op>                                                     \
    static bool                                                                \
    call(Op* op, const Tensor& tensor, int task_id, int num_tasks) {           \
      return call<Op>(op, tensor.dtype(), task_id, num_tasks);                 \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob, int task_id, int num_tasks) {   \
      return call<Op>(op, blob.meta(), task_id, num_tasks);                    \
    }                                                                          \
  };
C10_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes,
    DoRunWithType,
    DoRunWithOtherType)
C10_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes2,
    DoRunWithType2,
    DoRunWithOtherType2)
#undef C10_DEFINE_TENSOR_TYPES_DISPATCHER

} // namespace intra_op_parallel

} // namespace caffe2
