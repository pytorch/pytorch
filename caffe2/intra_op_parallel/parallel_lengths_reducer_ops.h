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

#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/intra_op_parallel/intra_op_parallel.h"
#include "caffe2/intra_op_parallel/partition.h"
#include "caffe2/perfkernels/embedding_lookup.h"

namespace caffe2 {

namespace intra_op_parallel {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    typename T, // output type
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = 0, // Whether this is SparseLengthsMean
    bool USE_POSITIONAL_WEIGHT = 0
    // USE_WEIGHT = 1 and USE_POSITIONAL_WEIGHT = 1
    // -> SparseLengthsPositionalWeightedSum
    >
class ParallelSparseLengthsReductionOp : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DISPATCH_HELPER;
  ParallelSparseLengthsReductionOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : ParallelOpBase(operator_def, ws) {
    static_assert(
        !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
  }

  ~ParallelSparseLengthsReductionOp() override {}

 protected:
  bool RunOnDevicePrologue(int /* unused */) override {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t M = lengthsInput.size(0);
    const int64_t indices_size = indicesInput.numel();

    auto shape = dataInput.sizes().vec();
    shape[0] = M;
    auto* output = Output(0, shape, at::dtype<T>());
    output->template mutable_data<T>();

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
      if (!USE_POSITIONAL_WEIGHT) {
        CAFFE_ENFORCE_EQ(
            weightInput.numel(),
            indices_size,
            "Weight should have the same length as indices.");
      }
    }

    return true;
  }

  // Currently, we support float and at::Half inputs for input data type, and
  // int32_t and int64_t for the index type.

  bool RunOnDeviceParallel(int task_id, int num_tasks) override {
    return DispatchHelper<InputTypes>::call(
        this, Input(DATA), task_id, num_tasks);
  }

  template <typename InputType>
  bool DoRunWithType(int task_id, int num_tasks) {
    return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
        this, Input(INDICES), task_id, num_tasks);
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2(int task_id, int num_tasks) {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    const int64_t N = dataInput.size(0);
    const int D = dataInput.size_from_dim(1);
    const int64_t M = lengthsInput.size(0);

    auto* output = Output(0);
    T* out_data = output->template mutable_data<T>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      in_weight = weightInput.template data<T>();
    }

    // delegate work to perfkernel that branches based on architecture
    size_t M_begin, M_end;
    std::tie(M_begin, M_end) = Get1DPartition(M, num_tasks, task_id);

    int indices_offset = std::accumulate(lengths, lengths + M_begin, 0);
    int64_t indices_size_of_tid =
        std::accumulate(lengths + M_begin, lengths + M_end, 0);
    const T* in_weight_of_tid = in_weight == nullptr
        ? nullptr
        : (USE_POSITIONAL_WEIGHT ? in_weight : in_weight + indices_offset);

    EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
        D,
        M_end - M_begin,
        indices_size_of_tid,
        N,
        in_data,
        indices + indices_offset,
        lengths + M_begin,
        in_weight_of_tid,
        nullptr, // scale_bias field is only used in
                 // SparseLengths8BitsRowwiseOp
        USE_MEAN,
        out_data + M_begin * D);
    return true;
  }

 private:
  enum {
    DATA = 0, // Data input.
    WEIGHT = 1, // Weight input used in SparseLengthsWeightedSum
    INDICES = 1 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
                              // 2 in SparseLengthsWeightedSum
    LENGTHS = 2 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
                              // 3 in SparseLengthsWeightedSum
  };
};

} // namespace intra_op_parallel

} // namespace caffe2
