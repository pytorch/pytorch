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

#include "tbb_numa_all_reduce_op.h"

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB

#include <array>
#include <utility>
#include <vector>

#include "intra_op_parallel.h"
#include "numa_all_reduce_op.h"
#include "numa_all_reduce_op_avx2.h"

// #define NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN

#ifdef NUMA_ALL_REDUCE_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

using namespace std;
using namespace ::tbb::flow;

C10_DECLARE_int(caffe2_intra_op_parallel_max_num_tasks);
C10_DECLARE_int(caffe2_intra_op_parallel_max_num_workers);

namespace caffe2 {

namespace tbb {

namespace {

// TODO: reuse the code in net_tbb_task_graph.cc
class pinning_observer : public ::tbb::task_scheduler_observer {
 public:
  pinning_observer(::tbb::task_arena& arena, int numa_node_id)
      : ::tbb::task_scheduler_observer(arena), numa_node_id_(numa_node_id) {
    observe(true);
  } // activate the observer

  void on_scheduler_entry(bool /* unused */) override {
    NUMABind(numa_node_id_);
  }

 private:
  int numa_node_id_;
};

// TODO: reuse the code in net_tbb_task_graph.cc
template <typename T>
std::unique_ptr<graph_node>
make_crossgraph_edge(sender<T>& s, receiver<T>& r, graph& receiver_g) {
  typedef async_node<T, T> async_node_t;
  auto node = new async_node_t(
      receiver_g,
      unlimited,
      [](T msg, typename async_node_t::gateway_type& gw) { gw.try_put(msg); });
  make_edge(s, *node);
  make_edge(*node, r);
  return std::unique_ptr<graph_node>(node);
}

} // namespace

NUMAAllReduceOp::NUMAAllReduceOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CPUContext>(operator_def, ws) {
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

NUMAAllReduceOp::~NUMAAllReduceOp() {
  for (auto i = 0; i < push_bufs_.size(); ++i) {
    free(push_bufs_[i]);
  }
}

// number of fp32 per cache line
constexpr int CACHE_LINE_LEN = 64 / sizeof(float);

namespace {

class ReduceScatter {
 public:
  /**
   * Reduce num_inputs vectors each with length len, and we are using num_tasks
   * for each input and each reduce-scatter step.
   * Tasks working on input_id'th input adds data received by my_push_buf to
   * its buffer, and passes the accumulated data to next_push_buf .
   *
   * idx_in_ring specifies the position of input_id in ring. The numbering for
   * input_id and idx_in_ring are often different depending on the topology.
   * For example, see rings variable defined in numa_all_reduce_op for mapping
   * we're using for twisted hyper-cube.
   */
  ReduceScatter(
      int step,
      int input_id,
      int task_id,
      int idx_in_ring,
      int num_inputs,
      int num_tasks,
      NUMAAllReduceOp* op,
      const float* my_push_buf,
      float* next_push_buf)
      : step_(step),
        input_id_(input_id),
        task_id_(task_id),
        idx_in_ring_(idx_in_ring),
        num_inputs_(num_inputs),
        num_tasks_(num_tasks),
        op_(op),
        my_push_buf_(my_push_buf),
        next_push_buf_(next_push_buf) {}

  void operator()() const {
    size_t len = op_->Output(input_id_)->numel();

    // TODO: we may want to support a case num_inputs_ can dynamically
    // change over time.

    // Partition the array for all-reduce into num_inputs chunks while
    // making each chunk is cache line aligned.
    size_t chunk_len = (len + num_inputs_ * CACHE_LINE_LEN - 1) / num_inputs_ /
        CACHE_LINE_LEN * CACHE_LINE_LEN;
    // Each chunk is in turn partitioned among tasks while making the portion
    // for each task is cache line aligned.
    size_t len_per_task = (chunk_len + num_tasks_ * CACHE_LINE_LEN - 1) /
        num_tasks_ / CACHE_LINE_LEN * CACHE_LINE_LEN;

    // we partition the array into num_inputs chunks
    // at ith step, socket s accumulates (s - i + num_inputs)th chunk and
    // pushes to socket s + 1
    int chunk = (idx_in_ring_ - step_ + num_inputs_) % num_inputs_;
    size_t chunk_begin = std::min(chunk * chunk_len, len);
    size_t chunk_end = std::min(chunk_begin + chunk_len, len);

    size_t task_begin =
        std::min(chunk_begin + task_id_ * len_per_task, chunk_end);
    size_t task_end = std::min(task_begin + len_per_task, chunk_end);

    float* my_buf = op_->Output(input_id_)->mutable_data<float>();

    if (step_ > 0) {
      // Accumulate wgt grads pushed from previous step
      // Note the difference from the implementation in numa_all_reduce that
      // doesn't use TBB. In the implementation there, stream_add belongs to
      // previous step. In TBB implementation, moving stream_add here aligns
      // with the synchronization we need among the tasks.
      // That is, stream_add has a dependency from stream_copy done in another
      // numa node.
      intra_op_parallel::stream_add(
          my_buf + task_begin,
          my_push_buf_ + task_begin,
          task_end - task_begin);
    }

    // Push to a buffer using non-temporal (streaming) store
    // Using non-temporal store here is very important to reduce coherency
    // overhead by avoiding shared cache line states.
    intra_op_parallel::stream_copy(
        next_push_buf_ + task_begin,
        my_buf + task_begin,
        task_end - task_begin);

    // Make sure non-temporal stores are fully visible to other threads
    _mm_sfence();
  }

  void operator()(continue_msg) {
    ReduceScatter::operator()();
  }

 private:
  int step_, input_id_, task_id_, idx_in_ring_, num_inputs_, num_tasks_;
  NUMAAllReduceOp* op_;
  const float* my_push_buf_;
  float* next_push_buf_;
}; // ReduceScatter

class AllGather {
 public:
  AllGather(
      int step,
      int input_id,
      int task_id,
      int idx_in_ring,
      int num_inputs,
      int num_tasks,
      NUMAAllReduceOp* op,
      const float* my_push_buf,
      int next_input_id)
      : step_(step),
        input_id_(input_id),
        task_id_(task_id),
        idx_in_ring_(idx_in_ring),
        num_inputs_(num_inputs),
        num_tasks_(num_tasks),
        op_(op),
        my_push_buf_(my_push_buf),
        next_input_id_(next_input_id) {}

  void operator()() const {
    size_t len = op_->Output(input_id_)->numel();

    // Partition the array for all-reduce into num_inputs chunks while
    // making each chunk is cache line aligned.
    size_t chunk_len = (len + num_inputs_ * CACHE_LINE_LEN - 1) / num_inputs_ /
        CACHE_LINE_LEN * CACHE_LINE_LEN;
    // Each chunk is in turn partitioned among tasks while making the portion
    // for each task is cache line aligned.
    size_t len_per_task = (chunk_len + num_tasks_ * CACHE_LINE_LEN - 1) /
        num_tasks_ / CACHE_LINE_LEN * CACHE_LINE_LEN;

    int chunk = (idx_in_ring_ + 1 - step_ + num_inputs_) % num_inputs_;
    size_t chunk_begin = std::min(chunk * chunk_len, len);
    size_t chunk_end = std::min(chunk_begin + chunk_len, len);

    size_t task_begin =
        std::min(chunk_begin + task_id_ * len_per_task, chunk_end);
    size_t task_end = std::min(task_begin + len_per_task, chunk_end);

    float* my_buf = op_->Output(input_id_)->mutable_data<float>();
    float* next_buf = op_->Output(next_input_id_)->mutable_data<float>();

    if (step_ == 0) {
      // This is technically the last step of reduce-scatter but moved to here
      // to reduce the number of tasks hence the number of synchronizations.
      intra_op_parallel::stream_add(
          my_buf + task_begin,
          my_push_buf_ + task_begin,
          task_end - task_begin);
    }

    intra_op_parallel::stream_copy(
        next_buf + task_begin, my_buf + task_begin, task_end - task_begin);

    // Make sure non-temporal stores are fully visible to other threads
    _mm_sfence();
  }

  void operator()(continue_msg) {
    AllGather::operator()();
  }

 private:
  int step_, input_id_, task_id_, idx_in_ring_, num_inputs_, num_tasks_;
  NUMAAllReduceOp* op_;
  const float* my_push_buf_;
  int next_input_id_;
}; // AllGather

// using intra_op_parallel::rings;

const std::vector<std::array<int, 8>> rings = {
    {0, 1, 2, 4, 7, 6, 5, 3},
    {0, 3, 5, 6, 7, 4, 2, 1},
};

void get_my_ring_info(
    int numa_node_id,
    int task,
    int num_numa_nodes,
    int* idx_in_ring,
    int* prev_sid,
    int* next_sid) {
  int ring_to_use = task % rings.size();
  *idx_in_ring =
      std::find(
          rings[ring_to_use].begin(), rings[ring_to_use].end(), numa_node_id) -
      rings[ring_to_use].begin();
  assert(*idx_in_ring != -1);
  *prev_sid =
      rings[ring_to_use][(*idx_in_ring - 1 + num_numa_nodes) % num_numa_nodes];
  *next_sid = rings[ring_to_use][(*idx_in_ring + 1) % num_numa_nodes];
  if (num_numa_nodes < 8) {
    *idx_in_ring = numa_node_id;
    *prev_sid = (numa_node_id - 1 + num_numa_nodes) % num_numa_nodes;
    *next_sid = (numa_node_id + 1) % num_numa_nodes;
    if (ring_to_use) {
      assert(rings.size() == 2);
      std::swap(*prev_sid, *next_sid);
    }
  }
}

} // namespace

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

  int num_tasks = 1;
  ExecutorHelper* executor_helper = OperatorBase::GetExecutorHelper();
  if (executor_helper) {
    TaskThreadPoolBase* pool = executor_helper->GetPool(device_option());
    assert(pool);

    num_tasks = std::min<int>(max_num_tasks_, pool->size());
    LOG(INFO) << pool->size() << " " << num_tasks;
  }

  // NUMA-aware allocation of push bufs
  // TODO: we may want to support a case that num_inputs and len can dynamically
  // change.
  if (push_bufs_.empty() && num_inputs > 1) {
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

    for (int numa_node_id = 0;
         numa_node_id < std::min(num_numa_nodes, num_inputs);
         ++numa_node_id) {
      arena_.emplace_back(new ::tbb::task_arena(num_tasks, numa_node_id == 0));
      observers_.emplace_back(
          new pinning_observer(*arena_[numa_node_id], numa_node_id));
    }

    dags_.resize(num_inputs);
    for (int input_id = 0; input_id < num_inputs; ++input_id) {
      dags_[input_id].reset(new graph);
      int numa_node_id = input_id % num_numa_nodes;
      arena_[numa_node_id]->execute(
          [this, input_id] { dags_[input_id]->reset(); });
    }

    dag_root_.reset(new cn_type(
        *dags_[0], [this](continue_msg) { dags_[0]->reserve_wait(); }));
    dag_exit_.reset(new cn_type(
        *dags_[0], [this](continue_msg) { dags_[0]->release_wait(); }));

    // reduce scatter
    for (int step = 0; step < num_inputs - 1; ++step) {
      for (int input_id = 0; input_id < num_inputs; ++input_id) {
        for (int task = 0; task < num_tasks; ++task) {
          int idx_in_ring, prev_input_id, next_input_id;
          get_my_ring_info(
              input_id,
              task,
              num_inputs,
              &idx_in_ring,
              &prev_input_id,
              &next_input_id);

          flow_nodes_.emplace_back(new cn_type(
              *dags_[input_id],
              ReduceScatter(
                  step,
                  input_id,
                  task,
                  idx_in_ring,
                  num_inputs,
                  num_tasks,
                  this,
                  push_bufs_[input_id],
                  push_bufs_[next_input_id])));

          if (step == 0) {
            // In the first step, add dependencies from the dag root
            if (input_id == 0) {
              make_edge(*dag_root_, *flow_nodes_.back());
            } else {
              cross_graph_edges_.push_back(make_crossgraph_edge(
                  *dag_root_, *flow_nodes_.back(), *dags_[input_id]));
            }
          } else {
            // Inter-numa-node dependency from previous step reduce scatter
            // flow_nodes_ is conceptually 3 dimensional array,
            // num_steps x num_inputs x num_tasks
            make_edge(
                *flow_nodes_
                    [((step - 1) * num_inputs + prev_input_id) * num_tasks +
                     task],
                *flow_nodes_.back());
            // FIXME: we're seeing deadlocks if we use the code below instead of
            // make_edge
            // cross_graph_edges_.push_back(make_crossgraph_edge(
            //     *flow_nodes_
            //         [((step - 1) * num_inputs + prev_input_id) * num_tasks +
            //          task],
            //     *flow_nodes_.back(),
            //     *dags_[input_id]));
          }
        } // for each task
      } // for each input
    } // for each reduce scatter step

    // all gather
    for (int step = 0; step < num_inputs - 1; ++step) {
      for (int input_id = 0; input_id < num_inputs; ++input_id) {
        for (int task = 0; task < num_tasks; ++task) {
          int idx_in_ring, prev_input_id, next_input_id;
          get_my_ring_info(
              input_id,
              task,
              num_inputs,
              &idx_in_ring,
              &prev_input_id,
              &next_input_id);

          flow_nodes_.emplace_back(new cn_type(
              *dags_[input_id],
              AllGather(
                  step,
                  input_id,
                  task,
                  idx_in_ring,
                  num_inputs,
                  num_tasks,
                  this,
                  push_bufs_[input_id],
                  next_input_id)));

          // Cross numa-node dependency from previous step
          make_edge(
              *flow_nodes_
                  [((num_inputs - 1 + step - 1) * num_inputs + prev_input_id) *
                       num_tasks +
                   task],
              *flow_nodes_.back());
          // FIXME: we're seeing deadlocks if we use the code below instead of
          // make_edge
          // cross_graph_edges_.push_back(make_crossgraph_edge(
          //     *flow_nodes_
          //         [((num_inputs - 1 + step - 1) * num_inputs + prev_input_id)
          //         *
          //              num_tasks +
          //          task],
          //     *flow_nodes_.back(),
          //     *dags_[input_id]));
          if (step == num_inputs - 2) {
            // In the last step, add dependencies to the dag exit
            if (input_id == 0) {
              make_edge(*flow_nodes_.back(), *dag_exit_);
            } else {
              cross_graph_edges_.push_back(make_crossgraph_edge(
                  *flow_nodes_.back(), *dag_exit_, *dags_[0]));
            }
          }
        } // for each task
      } // for each input
    } // for each all-gather step
  } // push_bufs_.empty()

  if (num_inputs > 1) {
    arena_[0]->execute([this] { dag_root_->try_put(continue_msg()); });
    arena_[0]->execute([this] { dags_[0]->wait_for_all(); });
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(NUMAAllReduce, TBB, NUMAAllReduceOp);

} // namespace tbb

} // namespace caffe2

#endif // INTRA_OP_PARALLEL_CAN_USE_TBB
