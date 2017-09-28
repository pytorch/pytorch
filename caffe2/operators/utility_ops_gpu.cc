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

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reshape_op.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

template <>
bool WeightedSumOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float>();
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
bool SumOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16, float16>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
class CopyOnDeviceLikeOp<CUDAContext, CUDAContext, CUDAContext>
    : public Operator<CUDAContext> {
 public:
  CopyOnDeviceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor<CUDAContext>>(0);
    CUDAContext context(GetGPUIDForPointer(Input(1).raw_data()));
    output->ResizeLike(input);
    context.template CopyItems<CUDAContext, CUDAContext>(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

REGISTER_CUDA_OPERATOR(Print, PrintOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Flatten, FlattenOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(FlattenToVec, FlattenToVecOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Alias, AliasOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeLike, ResizeLikeOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Sum, SumOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(WeightedSum, WeightedSumOp<CUDAContext>);
// From whatever the current context, ensure the output is TensorCPU
REGISTER_CUDA_OPERATOR(
    EnsureCPUOutput,
    CopyOp<CUDAContext, CPUContext, CUDAContext>);
// From CPU, copy it to whatever the current context
REGISTER_CUDA_OPERATOR(
    CopyFromCPUInput,
    CopyOp<CUDAContext, CUDAContext, CPUContext>);

// CopyGPUToCPU and CopyCPUToGPU should both be carried out in a cuda context,
// since gpu code will be involved.
REGISTER_CUDA_OPERATOR(
    CopyGPUToCPU,
    CopyOp<CUDAContext, CPUContext, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CopyCPUToGPU,
    CopyOp<CUDAContext, CUDAContext, CPUContext>);
// If we only specify Copy, we assume that it is a gpu to gpu copy - maybe
// involving different GPUs.
REGISTER_CUDA_OPERATOR(Copy, CopyOp<CUDAContext, CUDAContext, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CUDAContext, CUDAContext, CUDAContext>);

REGISTER_CUDA_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<CUDAContext>);

} // namespace caffe2
