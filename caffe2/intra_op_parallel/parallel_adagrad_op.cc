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

#include "parallel_adagrad_op.h"

#include "caffe2/perfkernels/adagrad.h"
#include "partition.h"

/// #define INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
namespace chrono = std::chrono;
#endif

namespace caffe2 {

namespace intra_op_parallel {

template <typename T>
ParallelAdagradOp<T>::ParallelAdagradOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : ParallelOpBase(operator_def, ws),
      epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f)),
      decay_(this->template GetSingleArgument<T>("decay", 1.0f)) {}

template <typename T>
bool ParallelAdagradOp<T>::RunOnDevicePrologue(int /* unused */) {
  CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENT_1).numel());
  CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(PARAM).numel());
  Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
  Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

  Output(OUTPUT_PARAM)->template mutable_data<T>();
  Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

  return true;
}

template <typename T>
bool ParallelAdagradOp<T>::RunOnDeviceParallel(int task_id, int num_tasks) {
  const T* param = Input(PARAM).template data<T>();
  const T* grad = Input(GRAD).template data<T>();
  const T* moment = Input(MOMENT_1).template data<T>();
  T* out_param = Output(OUTPUT_PARAM)->template mutable_data<T>();
  T* out_moment = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
  const T* lr = Input(LR).template data<T>();

#ifdef TBB_OP_PARALLEL_LOG_NUMA_NODE_AND_THREAD_ID
  if (use_tbb_) {
    LOG_FIRST_N(INFO, 512) << "task_id " << task_id << " numa_node_id "
                           << GetCurrentNUMANode() << " tid "
                           << ::tbb::this_task_arena::current_thread_index();
  }
#endif

  std::size_t work_begin, work_end;
  std::tie(work_begin, work_end) =
      Get1DPartition(Input(GRAD).numel(), num_tasks, task_id);

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  t_begin = chrono::system_clock::now();
#endif

  adagrad_update(
      work_end - work_begin,
      param + work_begin,
      grad + work_begin,
      moment + work_begin,
      out_param + work_begin,
      out_moment + work_begin,
      epsilon_,
      decay_,
      *lr);

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double adagrad_flops = 10.0;
  double dt = chrono::duration<double>(t_end - t_begin).count();
  double flops = (work_end - work_begin) * adagrad_flops;
  // if (0 == task_id) {
  LOG_FIRST_N(INFO, 512) << "Intra parallel Adagrad Op " << debug_def().input(1)
                         << " task_id " << task_id << " num_tasks " << num_tasks
                         << " Begin " << work_begin << " End " << work_end
                         << " " << dt << " sec " << flops << " flops "
                         << flops / num_tasks / dt / 1e9 << " GF/s/thread";
  // }

  t_begin = chrono::system_clock::now();
#endif

  return true;
}

template <typename T>
bool ParallelSparseAdagradOp<T>::RunOnDevicePrologue(int /* unused */) {
  // Enforce shapes
  CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
  CAFFE_ENFORCE_EQ(
      Input(PARAM).size_from_dim(1),
      Input(GRAD).size_from_dim(Input(INDICES).dim()));

  Output(OUTPUT_PARAM)->template mutable_data<T>();
  Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }

  // Enforce:
  // input(embedding/momentum) == outputs(embedding/momentum)
  CAFFE_ENFORCE_EQ(
      Input(PARAM).numel(),
      Input(MOMENT_1).numel(),
      "Input Param size: ",
      Input(PARAM).numel(),
      " Input Moment size: ",
      Input(MOMENT_1).numel());

  // input(grad) is compatible with size of indexes
  CAFFE_ENFORCE_EQ(
      Input(GRAD).numel() % n,
      0,
      "Incorrect gradient size:",
      Input(GRAD).numel(),
      " size of indexes:",
      n);

  return true;
}

template <typename T>
bool ParallelSparseAdagradOp<T>::RunOnDeviceParallel(
    int task_id,
    int num_tasks) {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, Input(INDICES), task_id, num_tasks);
}

template <typename T>
template <typename SIndex>
bool ParallelSparseAdagradOp<T>::DoRunWithType(int task_id, int num_tasks) {
  const auto* lr = Input(LR).template data<T>();
  const auto* indices = Input(INDICES).template data<SIndex>();
  const auto* gradIn = Input(GRAD).template data<T>();
  const auto* paramIn = Input(PARAM).template data<T>();
  const auto* momentIn = Input(MOMENT_1).template data<T>();
  auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
  auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }

#ifdef TBB_OP_LOG_NUMA_NODE_AND_THREAD_ID
  if (use_tbb_) {
    LOG_FIRST_N(INFO, 512) << "task_id " << task_id << " numa_node_id "
                           << GetCurrentNUMANode() << " tid "
                           << ::tbb::this_task_arena::current_thread_index();
  }
#endif

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  t_begin = chrono::system_clock::now();
#endif

  auto block_size = Input(GRAD).numel() / n;

  std::size_t work_begin, work_end;
  std::tie(work_begin, work_end) = Get1DPartition(n, num_tasks, task_id);

  int num_rows_processed = sparse_adagrad(
      work_end - work_begin,
      block_size,
      Input(PARAM).numel(),
      paramIn,
      gradIn + work_begin * block_size,
      momentIn,
      indices + work_begin,
      paramOut,
      momentOut,
      epsilon_,
      lr[0]);
  if (num_rows_processed < work_end - work_begin) {
    CAFFE_ENFORCE_GE(
        Input(PARAM).numel(),
        (indices[work_begin + num_rows_processed] + 1) * block_size,
        this->debug_def().input(PARAM),
        ", out of bound,  idx:",
        indices[work_begin + num_rows_processed],
        " for input i:",
        work_begin + num_rows_processed,
        " and block_size:",
        block_size,
        " max size:",
        Input(PARAM).numel());
    return false;
  }

#ifdef INTRA_OP_PARALLEL_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double adagrad_flops = 10.0;
  double dt = chrono::duration<double>(t_end - t_begin).count();
  double flops = (work_end - work_begin) * adagrad_flops;
  // if (0 == task_id) {
  LOG_FIRST_N(INFO, 512) << "Intra parallel Adagrad Op " << debug_def().input(1)
                         << " task_id " << task_id << " num_tasks " << num_tasks
                         << " Begin " << work_begin << " End " << work_end
                         << " " << dt << " sec " << flops << " flops "
                         << flops / num_tasks / dt / 1e9 << " GF/s/thread";
  // }

  t_begin = chrono::system_clock::now();
#endif

  return true;
}

namespace {

template <typename T>
class ParallelRowWiseSparseAdagradOp final : public ParallelOpBase {
 public:
  ParallelRowWiseSparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : ParallelOpBase(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevicePrologue(int /* unused */) override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    Output(OUTPUT_PARAM)->template mutable_data<T>();
    Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).numel() / n;

    // Enforce:
    // Input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel() / block_size,
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Block size: ",
        block_size,
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    // input(grad) is compatible with size of indexes
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel() % n,
        0,
        "Incorrect gradient size:",
        Input(GRAD).numel(),
        " size of indexes:",
        n);

    return true;
  }

  bool RunOnDeviceParallel(int task_id, int num_tasks) override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES), task_id, num_tasks);
  }

  template <typename SIndex>
  bool DoRunWithType(int task_id, int num_tasks) {
    const auto* lr = Input(LR).template data<T>();
    auto* param = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).numel() / n;

    std::size_t work_begin, work_end;
    std::tie(work_begin, work_end) = Get1DPartition(n, num_tasks, task_id);

    for (int i = work_begin; i < work_end; ++i) {
      std::size_t idx = indices[i];
      auto offsetI = i * block_size;
      auto offsetIdx = idx * block_size;

      // Enforce:
      // access within range
      // gradient access within range
      CAFFE_ENFORCE_GE(
          Input(PARAM).numel(),
          block_size + offsetIdx,
          this->debug_def().input(PARAM),
          ", out of bound,  idx:",
          idx,
          " for input i:",
          i,
          " and block size:",
          block_size,
          " max size:",
          Input(PARAM).numel());

      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = moment[idx] = moment[idx] + gi * gi;
        param[idx] = param[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
      } else {
        // prefetching
        const int prefdist_T0 = 16;
        int i_pref = (i < n - prefdist_T0) ? i + prefdist_T0 : i;
        std::size_t idx_pref = indices[i_pref];

        internal::rowwise_adagrad_update_inlined(
            block_size,

            param + offsetIdx,
            &param[idx_pref * block_size],

            gradIn + offsetI,

            moment + idx,
            moment + idx_pref,

            epsilon_,
            lr[0]);
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

} // namespace

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Adagrad,
    INTRA_OP_PARALLEL,
    ParallelAdagradOp<float>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseAdagrad,
    INTRA_OP_PARALLEL,
    ParallelSparseAdagradOp<float>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagrad,
    INTRA_OP_PARALLEL,
    ParallelRowWiseSparseAdagradOp<float>);

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB
REGISTER_CPU_OPERATOR_WITH_ENGINE(Adagrad, TBB, ParallelAdagradOp<float>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SparseAdagrad,
    TBB,
    ParallelSparseAdagradOp<float>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagrad,
    TBB,
    ParallelRowWiseSparseAdagradOp<float>);
#endif

} // namespace intra_op_parallel

} // namespace caffe2
