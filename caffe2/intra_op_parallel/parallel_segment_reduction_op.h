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
#include "caffe2/operators/reducer_functors.h"

namespace caffe2 {

namespace intra_op_parallel {

/*
 * Some notice:
 * 1. Gradient actually doesn't depend on whether sparse lookup is fused or not
 * 2. INDICES are not used in CPU version, but they are needed in async CUDA
 *    version. So we register 3 input version for CPU as gradient op for
 *    GPU/CPU convert. We then register 2 input version for CPU for backward
 *    compatibility with older nets.
 */
template <
    typename T,
    typename TLengths,
    class ReducerGradient,
    bool GradientNeedIndices = false>
class ParallelAbstractLengthsGradientOp : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DISPATCH_HELPER;
  ParallelAbstractLengthsGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : ParallelOpBase(operator_def, ws) {}
  virtual ~ParallelAbstractLengthsGradientOp() noexcept override {}

 protected:
  bool RunOnDevicePrologue(int /* unused */) override {
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t reducedDataSize = 0;
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
    const TLengths* lengths = lengthsInput.template data<TLengths>();
    for (int64_t i = 0; i < numSegments; ++i) {
      reducedDataSize += lengths[i];
    }

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          reducedDataSize,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
    }

    vector<int64_t> shape;
    shape.push_back(reducedDataSize);
    ctx.appendGradShape(&shape);
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    dataGradsOutput->template mutable_data<T>();

    return true;
  }

  bool RunOnDeviceParallel(int task_id, int num_tasks) override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t gradBlockSize = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, gradBlockSize, task_id, num_tasks);
  }

  template <int FixedSize>
  bool DoRunWithValue(int task_id, int num_tasks) {
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);
    auto* dataGradsOutput = Output(0);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t reducedDataSize = 0;
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
    const TLengths* lengths = lengthsInput.template data<TLengths>();
    for (int64_t i = 0; i < numSegments; ++i) {
      reducedDataSize += lengths[i];
    }

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
    }

    const T* segmentGrads = segmentGradsInput.template data<T>();

    vector<int64_t> shape;
    shape.push_back(reducedDataSize);
    ctx.appendGradShape(&shape);

    int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
    int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
    T* dataGrads = dataGradsOutput->template mutable_data<T>();

    size_t rangeIndexBegin, rangeIndexEnd;
    std::tie(rangeIndexBegin, rangeIndexEnd) =
        Get1DPartition(numSegments, num_tasks, task_id);
    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      if (rangeIndex >= rangeIndexBegin && rangeIndex < rangeIndexEnd) {
        ReducerGradient reducer(
            ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
        for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
             ++dataIndex) {
          reducer.template fillGrad<FixedSize>(
              ctx,
              dataGrads + dataGradsBlockSize * dataIndex,
              dataIndex,
              &context_,
              lengths[rangeIndex]);
        }
      } else {
        dataIndex += lengths[rangeIndex];
      }
    }
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS, INDICES
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs = ReducerGradient::originalInputs().size() +
      2 + (GradientNeedIndices ? 1 : 0);
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    LENGTHS,
    INDICES
  };
};

// Version of gradient that requires the main input and thus needs to receive
// length, indices and other stuff
template <
    typename T,
    typename TLengths,
    class ReducerGradient,
    bool SparseFused = true,
    bool GradientNeedIndices = false>
class ParallelAbstractLengthsWithMainInputGradientOp : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DISPATCH_HELPER;
  ParallelAbstractLengthsWithMainInputGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : ParallelOpBase(operator_def, ws) {}
  virtual ~ParallelAbstractLengthsWithMainInputGradientOp() noexcept override {}

 protected:
  bool RunOnDevicePrologue(int /* unused */) override {
    auto& dataInput = Input(DATA_INPUT);
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      int aux_num = ReducerGradient::originalInputs()[i];
      auto& aux_in = Input(i);
      auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
      ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
    }

    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      dataToReduceSize = indicesInput.size(0);
    } else {
      dataToReduceSize = dataInput.size(0);
    }

    vector<int64_t> shape;
    shape.push_back(dataToReduceSize);
    ctx.appendGradShape(&shape);
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    dataGradsOutput->template mutable_data<T>();

    return true;
  }

  bool RunOnDeviceParallel(int task_id, int num_tasks) override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES), task_id, num_tasks);
    } else {
      // type doesn't matter
      return DoRunWithType<int64_t>(task_id, num_tasks);
    }
  }

  template <typename IndexType>
  bool DoRunWithType(int task_id, int num_tasks) {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch, IndexType>::
        call(this, in_block_size, task_id, num_tasks);
  }

  template <typename IndexType, int FixedSize>
  bool DoRunWithValue(int task_id, int num_tasks) {
    auto& dataInput = Input(DATA_INPUT);
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);
    auto* dataGradsOutput = Output(0);

    int64_t numSegments = lengthsInput.size(0);
    const TLengths* lengths = lengthsInput.template data<TLengths>();

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      int aux_num = ReducerGradient::originalInputs()[i];
      auto& aux_in = Input(i);
      auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
      ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
    }

    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const IndexType* indices = nullptr;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      indices = indicesInput.template data<IndexType>();
      dataToReduceSize = indicesInput.size(0);
    } else {
      dataToReduceSize = dataInput.size(0);
    }

    const T* segmentGrads = segmentGradsInput.template data<T>();

    vector<int64_t> shape;
    shape.push_back(dataToReduceSize);
    ctx.appendGradShape(&shape);

    int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
    int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
    T* dataGrads = dataGradsOutput->template mutable_data<T>();

    const T* data = dataInput.template data<T>();

    size_t rangeIndexBegin, rangeIndexEnd;
    std::tie(rangeIndexBegin, rangeIndexEnd) =
        Get1DPartition(numSegments, num_tasks, task_id);
    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      if (rangeIndex >= rangeIndexBegin && rangeIndex < rangeIndexEnd) {
        ReducerGradient reducer(
            ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
        for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
             ++dataIndex) {
          IndexType data_pos;
          // No range checking, should've been verified in forward pass
          if (SparseFused) { // static if
            data_pos = indices[dataIndex];
          } else {
            data_pos = dataIndex;
          }
          reducer.template fillGradWithMainInput<FixedSize>(
              ctx,
              data + dataGradsBlockSize * data_pos,
              dataGrads + dataGradsBlockSize * dataIndex,
              dataIndex,
              &context_,
              lengths[rangeIndex]);
        }
      } else {
        dataIndex += lengths[rangeIndex];
      }
    }
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS,
  //      DATA_INPUT, [INDICES]
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs = ReducerGradient::originalInputs().size() +
      3 + (SparseFused ? 1 : 0) + (GradientNeedIndices ? 1 : 0);
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    LENGTHS,
    DATA_INPUT,
    INDICES,
  };
};

} // namespace intra_op_parallel

} // namespace caffe2
