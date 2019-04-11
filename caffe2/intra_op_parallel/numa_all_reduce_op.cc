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

#include "numa_all_reduce_op.h"

#include <utility>

#include "intra_op_parallel.h"
#include "numa_all_reduce_op_avx2.h"

// #define NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

using namespace std;

C10_DECLARE_int(caffe2_intra_op_parallel_max_num_tasks);
C10_DECLARE_int(caffe2_intra_op_parallel_max_num_workers);
C10_DECLARE_bool(caffe2_intra_op_parallel_only_grab_idle_threads);

namespace caffe2 {

namespace intra_op_parallel {

// The 2 rings embedded in the twisted hyper-cube topology (ASCII art below)
// used in Intel 8-socket machines (there're also links 0-7 and 1-6 that I
// couldn't draw with ASCII art).
// 0 - 1
// |   |
// 3 - 2
//   X
// 4 - 5
// |   |
// 7 - 6
// TODO: make this more generic to perform well also in 4 sockets machines.
static const std::vector<std::array<int, 8>> rings = {
    {0, 1, 2, 4, 7, 6, 5, 3},
    {0, 3, 5, 6, 7, 4, 2, 1},
};

void get_my_ring_info(
    int numa_node_id,
    int task,
    int num_numa_nodes,
    int* idx_in_ring,
    int* prev_numa_node_id,
    int* next_numa_node_id) {
  // Threads in the same numa node use different rings in an interleaved way
  // to fully utilize all the outgoing UPI links from each numa node.
  int ring_to_use = task % rings.size();
  *idx_in_ring =
      std::find(
          rings[ring_to_use].begin(), rings[ring_to_use].end(), numa_node_id) -
      rings[ring_to_use].begin();
  assert(*idx_in_ring != -1);
  *prev_numa_node_id =
      rings[ring_to_use][(*idx_in_ring - 1 + num_numa_nodes) % num_numa_nodes];
  *next_numa_node_id = rings[ring_to_use][(*idx_in_ring + 1) % num_numa_nodes];
  if (num_numa_nodes < 8) {
    // When num_numa_nodes != 8, just assume there's a direct channel between
    // ith numa domain and (i + 1)th numa domain (with wrapping around).
    *idx_in_ring = numa_node_id;
    *prev_numa_node_id = (numa_node_id - 1 + num_numa_nodes) % num_numa_nodes;
    *next_numa_node_id = (numa_node_id + 1) % num_numa_nodes;
    if (ring_to_use) {
      std::swap(*prev_numa_node_id, *next_numa_node_id);
    }
  }
}

constexpr int MAX_NUM_THREADS = 1024;

NUMAAllReduceOp::NUMAAllReduceOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      count_(0),
      cv_for_peer_sync_(MAX_NUM_THREADS),
      mutex_for_peer_sync_(MAX_NUM_THREADS),
      generations_(MAX_NUM_THREADS) {
  for (int i = 0; i < MAX_NUM_THREADS; ++i) {
    generations_[i].reset(new atomic<int>);
  }

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
  // TODO: numa-aware allocation of cv_, mutex_, and generations_
}

NUMAAllReduceOp::~NUMAAllReduceOp() {
  for (auto i = 0; i < push_bufs_.size(); ++i) {
    free(push_bufs_[i]);
  }
}

// number of fp32 per cache line
constexpr int CACHE_LINE_LEN = 64 / sizeof(float);

// See http://research.baidu.com/bringing-hpc-techniques-deep-learning/
bool NUMAAllReduceOp::RunOnDeviceParallel_(
    int numa_node_id,
    int task_id,
    int num_tasks) {
#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_very_begin, t_begin, t_end;
  t_very_begin = chrono::system_clock::now();
#endif

  size_t len = Input(0).numel();
  int num_numa_nodes = InputSize();

  vector<float*> outputs(num_numa_nodes);
  for (int i = 0; i < num_numa_nodes; ++i) {
    outputs[i] = static_cast<float*>(Output(i)->mutable_data<float>());
  }

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
  t_begin = chrono::system_clock::now();
#endif

  // Threads in the same numa node use different rings in an interleaved way
  // to fully utilize all the outgoing UPI links from each numa node.
  int idx_in_ring, prev_numa_node_id, next_numa_node_id;
  get_my_ring_info(
      numa_node_id,
      task_id,
      num_numa_nodes,
      &idx_in_ring,
      &prev_numa_node_id,
      &next_numa_node_id);

  int my_global_task_id = numa_node_id * num_tasks + task_id;
  int next_global_task_id = next_numa_node_id * num_tasks + task_id;

  float* my_buf = outputs[numa_node_id];
  float* next_buf = outputs[next_numa_node_id];

  const float* my_push_buf = push_bufs_[numa_node_id];
  float* next_push_buf = push_bufs_[next_numa_node_id];

  // Partition the array for all-reduce into num_numa_nodes chunks while making
  // each chunk is cache line aligned.
  size_t chunk_len = (len + num_numa_nodes * CACHE_LINE_LEN - 1) /
      num_numa_nodes / CACHE_LINE_LEN * CACHE_LINE_LEN;
  // Each chunk is in turn partitioned among tasks while making the portion
  // for each task is cache line aligned.
  size_t len_per_task = (chunk_len + num_tasks * CACHE_LINE_LEN - 1) /
      num_tasks / CACHE_LINE_LEN * CACHE_LINE_LEN;
  int local_generation = 0;

  /////////////////////////////////////////////////////////////////////////////
  // ReduceScatter
  for (int step = 0; step < num_numa_nodes - 1; ++step, ++local_generation) {
    // At ith step, numa node s read (num_numa_nodes - 1 + s - i)th chunk from
    // numa node s - 1 and accumulates to its local chunk
    int chunk_to_push = (idx_in_ring - step + num_numa_nodes) % num_numa_nodes;

    size_t chunk_begin = std::min(chunk_to_push * chunk_len, len);
    size_t chunk_end = std::min(chunk_begin + chunk_len, len);

    size_t task_begin =
        std::min(chunk_begin + task_id * len_per_task, chunk_end);
    size_t task_end = std::min(task_begin + len_per_task, chunk_end);

    // Push to a buffer using non-temporal (streaming) store
    // Using non-temporal store here is very important to reduce coherency
    // overhead by avoiding shared cache line states.
    stream_copy(
        next_push_buf + task_begin, my_buf + task_begin, task_end - task_begin);

    {
      lock_guard<mutex> lock(mutex_for_peer_sync_[next_global_task_id]);
      ++(*generations_[next_global_task_id]);
    }
    cv_for_peer_sync_[next_global_task_id].notify_one();

    {
      unique_lock<mutex> lock(mutex_for_peer_sync_[my_global_task_id]);
      atomic<int>* generation_to_wait = generations_[my_global_task_id].get();
      cv_for_peer_sync_[my_global_task_id].wait(
          lock, [generation_to_wait, local_generation] {
            return *generation_to_wait > local_generation;
          });
    }

    int chunk_to_read = (chunk_to_push - 1 + num_numa_nodes) % num_numa_nodes;
    chunk_begin = std::min(chunk_to_read * chunk_len, len);
    chunk_end = std::min(chunk_begin + chunk_len, len);

    task_begin = std::min(chunk_begin + task_id * len_per_task, chunk_end);
    task_end = std::min(task_begin + len_per_task, chunk_end);

    // accumulate
    stream_add(
        my_buf + task_begin, my_push_buf + task_begin, task_end - task_begin);
  } // for each step of reduce-scatter

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
  if (0 == numa_node_id && 0 == task_id) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    double bytes = (double)sizeof(float) * len * (num_numa_nodes - 1);
    LOG(INFO) << "Reduce-scatter aggregate effective BW " << bytes / dt / 1e9
              << " GB/s";
  }
  t_begin = chrono::system_clock::now();
#endif

  /////////////////////////////////////////////////////////////////////////////
  // AllGather
  for (int step = 0; step < num_numa_nodes - 1; ++step, ++local_generation) {
    int chunk_to_push =
        (idx_in_ring + 1 - step + num_numa_nodes) % num_numa_nodes;

    size_t chunk_begin = std::min(chunk_to_push * chunk_len, len);
    size_t chunk_end = std::min(chunk_begin + chunk_len, len);

    size_t task_begin =
        std::min(chunk_begin + task_id * len_per_task, chunk_end);
    size_t task_end = std::min(task_begin + len_per_task, chunk_end);

    stream_copy(
        next_buf + task_begin, my_buf + task_begin, task_end - task_begin);

    {
      lock_guard<mutex> lock(mutex_for_peer_sync_[next_global_task_id]);
      ++(*generations_[next_global_task_id]);
    }
    cv_for_peer_sync_[next_global_task_id].notify_one();

    {
      unique_lock<mutex> lock(mutex_for_peer_sync_[my_global_task_id]);
      atomic<int>* generation_to_wait = generations_[my_global_task_id].get();
      cv_for_peer_sync_[my_global_task_id].wait(
          lock, [generation_to_wait, local_generation] {
            return *generation_to_wait > local_generation;
          });
    }
  } // for each step of all-gather

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
  if (0 == numa_node_id && 0 == task_id) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    double bytes = (double)sizeof(float) * len * (num_numa_nodes - 1);
    LOG(INFO) << "All-gather aggregate effective BW " << bytes / dt / 1e9
              << " GB/s";
  }
#endif

  if (++count_ == num_numa_nodes * num_tasks) {
    count_ = 0;
    event().SetFinished();

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_very_begin).count();
    double bytes = (double)sizeof(float) * len * (num_numa_nodes - 1);
    LOG(INFO) << "All-reduce aggregate effective BW " << 2 * bytes / dt / 1e9
              << " GB/s len";
#endif
  }

  return true;
}

bool NUMAAllReduceOp::RunOnDevice() {
  // Check constraints
  int num_inputs = InputSize();
  CAFFE_ENFORCE_GE(num_inputs, 1, "Must have at least one input");

  size_t len = Input(0).numel();
  for (int i = 1; i < num_inputs; ++i) {
    CAFFE_ENFORCE_EQ(
        Input(i).numel(), len, "All inputs must have the same size");
  }
  int num_numa_nodes = GetNumNUMANodes();

  // NUMA-aware allocation of push bufs
  if (push_bufs_.empty()) {
    push_bufs_.resize(num_inputs);

    for (int input_id = 0; input_id < num_inputs; ++input_id) {
      CAFFE_ENFORCE_EQ(
          posix_memalign(
              (void**)&push_bufs_[input_id], gAlignment, len * sizeof(float)),
          0);
      push_bufs_[input_id] = (float*)aligned_alloc(4096, len * sizeof(float));
      // It's possible that the number of inputs (the number of NUMA nodes
      // user thought) is different from the real number of NUMA nodes, so
      // we wrap around by taking a modular.
      NUMAMove(
          push_bufs_[input_id], len * sizeof(float), input_id % num_numa_nodes);
    }
  }

  // Agree on how many tasks we will use per NUMA domain
  ExecutorHelper* executor_helper = OperatorBase::GetExecutorHelper();
  // TODO: need to implement a fallback path if executor_helper is NULL
  CAFFE_ENFORCE(
      executor_helper,
      "Cannot access executor helper. Check if you're running with a correct "
      "net type.");

  int num_tasks_per_numa = max_num_tasks_;
  int op_numa_id = GetCurrentNUMANode();
  for (int input_id = 0; input_id < num_inputs; ++input_id) {
    DeviceOption device = device_option();
    device.set_numa_node_id(input_id % num_numa_nodes);
    TaskThreadPoolBase* pool = executor_helper->GetPool(device);
    num_tasks_per_numa = std::min<int>(
        num_tasks_per_numa,
        FLAGS_caffe2_intra_op_parallel_only_grab_idle_threads
            ? pool->numAvailable() + (op_numa_id == input_id ? 1 : 0)
            : pool->size());
    // When device_option().numa_node_id() is same as input_id, we can use
    // the thread that's already running this RunOnDevice function
  }
  num_tasks_per_numa = std::max<int>(1, num_tasks_per_numa);
  VLOG(2) << "Executing " << debug_def().type() << " with "
          << num_tasks_per_numa << " tasks per NUMA node";

  // Initialize synchronization variables
  for (auto i = 0; i < num_inputs * num_tasks_per_numa; ++i) {
    *generations_[i] = 0;
  }

  // Enqueue tasks
  for (int numa_id = 0; numa_id < std::min(num_inputs, num_numa_nodes);
       ++numa_id) {
    ParallelOpBase::TaskPoolMutex(numa_id).lock();
  }
  for (int input_id = 0; input_id < num_inputs; ++input_id) {
    DeviceOption device = device_option();
    device.set_numa_node_id(input_id % num_numa_nodes);
    TaskThreadPoolBase* pool = executor_helper->GetPool(device);
    for (int task_id = (op_numa_id == input_id ? 1 : 0);
         task_id < num_tasks_per_numa;
         ++task_id) {
      pool->run(bind(
          &NUMAAllReduceOp::RunOnDeviceParallel_,
          this,
          input_id,
          task_id,
          num_tasks_per_numa));
    }
  }
  for (int numa_id = 0; numa_id < std::min(num_inputs, num_numa_nodes);
       ++numa_id) {
    ParallelOpBase::TaskPoolMutex(numa_id).unlock();
  }

  if (op_numa_id >= 0 && op_numa_id < num_inputs) {
    RunOnDeviceParallel_(op_numa_id, 0, num_tasks_per_numa);
  }

  return true;
}

OPERATOR_SCHEMA(NUMAAllReduce)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

For a system with N NUMA nodes, N input tensors and N output tensors should be
provided.
All tensors are in-place. All-reduce will be done in-place for the N
input/output tensors.
All tensors should be NUMA-aware allocated. That is ith
input tensor should be allocated to the local memory of ith numa node.
Multiple workers in each numa domain participate the all-reduce
if enough work is available.

)DOC")
    .Arg(
        "max_num_workers",
        "The maximum number of workers per NUMA, default 8 "
        "(usually this is enough to saturate cross NUMA domain BW)");

REGISTER_CPU_OPERATOR(NUMAAllReduce, NUMAAllReduceOp);

SHOULD_NOT_DO_GRADIENT(NUMAAllReduce);

} // namespace intra_op_parallel

} // namespace caffe2
