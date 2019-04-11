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

#include "tbb_fully_connected_op.h"

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB

//#define INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
//#define TBB_OP_LOG_NUMA_NODE_AND_THREAD_ID

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#include <mkl.h>

#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

#include "caffe2/utils/math.h"
#include "partition.h"

C10_DECLARE_int(caffe2_intra_op_parallel_max_num_tasks);
C10_DECLARE_int(caffe2_intra_op_parallel_max_num_workers);

C10_DECLARE_double(caffe2_intra_op_parallel_fc_dx_gemm_aspect_ratio);
C10_DECLARE_double(caffe2_intra_op_parallel_fc_dw_gemm_aspect_ratio);

using namespace std;

namespace caffe2 {

namespace tbb {

static constexpr int MAX_NUM_THREADS = 1024;

TBBFullyConnectedGradientOp::TBBFullyConnectedGradientOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
      axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)) {
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

bool TBBFullyConnectedGradientOp::RunOnDevice() {
  // Prologue should allocate all output tensors.
  if (!RunOnDevicePrologue()) {
    return false;
  }

  // Make sure RunOnDevicePrologue has allocated all outputs with mutable_data.
  // This is because resizing of tensor that can be trigerred by mutable_data
  // is not thread safe.
  for (int i = 0; i < OutputSize(); ++i) {
    Output(i)->raw_mutable_data();
  }

  int num_tasks = 1;
  ExecutorHelper* executor_helper = OperatorBase::GetExecutorHelper();
  if (executor_helper) {
    TaskThreadPoolBase* pool = executor_helper->GetPool(device_option());
    assert(pool);

    num_tasks = std::min<int>(max_num_tasks_, pool->size());
  }

  ::tbb::parallel_for(
      0,
      num_tasks,
      [this, num_tasks](size_t task_id) {
        int old_mkl_num_threads = mkl_set_num_threads_local(1);
        ComputeDWParallel_(task_id, num_tasks);
        mkl_set_num_threads_local(old_mkl_num_threads);
      },
      ::tbb::simple_partitioner());

  ::tbb::parallel_for(
      0,
      num_tasks,
      [this, num_tasks](size_t task_id) {
        int old_mkl_num_threads = mkl_set_num_threads_local(1);
        ComputeDXParallel_(task_id, num_tasks);
        mkl_set_num_threads_local(old_mkl_num_threads);
      },
      ::tbb::simple_partitioner());
  return true;
}

bool TBBFullyConnectedGradientOp::RunOnDevicePrologue() {
  const auto& X = Input(0);
  const auto& W = Input(1);
  // batch size
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int M = X.size_to_dim(canonical_axis);
  const int K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);
  CAFFE_ENFORCE(M * K == X.numel());
  CAFFE_ENFORCE(K * N == W.numel());

  auto* dW = Output(0, W.sizes(), at::dtype<float>());
  auto* db = Output(1, {N}, at::dtype<float>());

  if (X.numel() == 0) {
    // generate a zero blob for db and dW when X is empty
    math::Set<float, CPUContext>(
        db->numel(), 0.0f, db->mutable_data<float>(), &context_);
    math::Set<float, CPUContext>(
        dW->numel(), 0.0f, dW->mutable_data<float>(), &context_);

    if (OutputSize() == 3) {
      auto* dX = Output(2, X.sizes(), at::dtype<float>());
      dX->mutable_data<float>();
    }

    return true;
  }

  dW->mutable_data<float>();
  db->mutable_data<float>();

  if (OutputSize() == 3) {
    auto* dX = Output(2, X.sizes(), at::dtype<float>());
    dX->mutable_data<float>();
  }

  return true;
}

bool TBBFullyConnectedGradientOp::ComputeDWParallel_(
    int task_id,
    int num_tasks) {
  const auto& X = Input(0);
  const auto& W = Input(1);
  const auto& dY = Input(2);
  // batch size
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int M = X.size_to_dim(canonical_axis);
  const int K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  auto* dW = Output(0);
  auto* db = Output(1);

#ifdef TBB_OP_LOG_NUMA_NODE_AND_THREAD_ID
  LOG_FIRST_N(INFO, 512) << "task_id " << task_id << " numa_node_id "
                         << GetCurrentNUMANode() << " tid "
                         << ::tbb::this_task_arena::current_thread_index();
#endif

  if (X.numel() == 0) {
    return true;
  }

  const float* X_data = X.data<float>();
  const float* dY_data = dY.data<float>();

  float* dW_data = dW->mutable_data<float>();
  float* db_data = db->mutable_data<float>();

  int n_begin, n_end, k_begin, k_end;
  intra_op_parallel::Get2DPartition(
      &n_begin,
      &n_end,
      &k_begin,
      &k_end,
      N,
      K,
      FLAGS_caffe2_intra_op_parallel_fc_dw_gemm_aspect_ratio,
      true /* m_align */,
      num_tasks,
      task_id);

  // Compute dW
#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  t_begin = chrono::system_clock::now();
#endif

  cblas_sgemm(
      CblasRowMajor,
      CblasTrans,
      CblasNoTrans,
      n_end - n_begin,
      k_end - k_begin,
      M,
      1.0f,
      dY_data + n_begin,
      N,
      X_data + k_begin,
      K,
      0.f,
      dW_data + K * n_begin + k_begin,
      K);

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  double flops = 2.0 * M * N * K;
  // if (0 == task_id) {
  LOG_FIRST_N(INFO, 512) << "FC dW GEMM with weight blob "
                         << debug_def().input(1) << " task_id " << task_id
                         << " M " << M << " N " << N << " K " << K << " " << dt
                         << " sec " << flops << " flops "
                         << flops / num_tasks / dt / 1e9 << " GF/s/thread";
  // }

  t_begin = chrono::system_clock::now();
#endif

  // Compute dB
  // [n_begin:n_end] is further decomposed among the threads that worked on
  // [n_begin:n_end] when computing dW
  int n_begin_for_db =
      std::min(n_begin + (k_begin * (n_end - n_begin) + K - 1) / K, n_end);
  int n_end_for_db =
      std::min(n_begin + (k_end * (n_end - n_begin) + K - 1) / K, n_end);
  for (int i = n_begin_for_db; i < n_end_for_db; ++i) {
    float sum = 0;
    for (int j = 0; j < M; ++j) {
      sum += dY_data[i + j * N];
    }
    db_data[i] = sum;
  }

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  // t_end = chrono::system_clock::now();
  // dt = chrono::duration<double>(t_end - t_begin).count();
  // if (0 == task_id) {
  //   LOG(INFO) << "FC dB update with weight blob " << debug_def().input(1)
  //             << " task_id " << task_id << " " << dt << " sec";
  // }
#endif

  return true;
}

void TBBFullyConnectedGradientOp::ComputeDXParallel_(
    int task_id,
    int num_tasks) {
  const auto& X = Input(0);
  const auto& W = Input(1);
  const auto& dY = Input(2);
  // batch size
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int M = X.size_to_dim(canonical_axis);
  const int K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  const float* dY_data = dY.data<float>();
  const float* W_data = W.data<float>();

  auto* dX = Output(2);
  float* dX_data = dX->mutable_data<float>();

#ifdef TBB_OP_LOG_NUMA_NODE_AND_THREAD_ID
  LOG_FIRST_N(INFO, 512) << "task_id " << task_id << " numa_node_id "
                         << GetCurrentNUMANode() << " tid "
                         << ::tbb::this_task_arena::current_thread_index();
#endif

  // Compute dX
#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  t_begin = chrono::system_clock::now();
#endif
  int m_begin, m_end, k_begin, k_end;

  intra_op_parallel::Get2DPartition(
      &m_begin,
      &m_end,
      &k_begin,
      &k_end,
      M,
      K,
      FLAGS_caffe2_intra_op_parallel_fc_dx_gemm_aspect_ratio,
      false /* m_align */,
      num_tasks,
      task_id);

  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      m_end - m_begin,
      k_end - k_begin,
      N,
      1.0f,
      dY_data + m_begin * N,
      N,
      W_data + k_begin,
      K,
      0.0f,
      dX_data + m_begin * K + k_begin,
      K);

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  double flops = 2.0 * M * N * K;
  // if (0 == task_id) {
  LOG_FIRST_N(INFO, 512) << "FC dX GEMM with weight blob "
                         << debug_def().input(1) << " task_id " << task_id
                         << " num_tasks " << num_tasks << " M " << M << " N "
                         << N << " K " << K << " " << dt << " sec " << flops
                         << " flops " << flops / num_tasks / dt / 1e9
                         << " GF/s/thread";
  // }
#endif
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(FCGradient, TBB, TBBFullyConnectedGradientOp);

} // namespace tbb

} // namespace caffe2

#endif // INTRA_OP_PARALLEL_CAN_USE_TBB
