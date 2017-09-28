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
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

template <>
bool FullyConnectedOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, // X
        float16, // W
        float16, // B
        float16, // Y
        float>(); // Math
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

template <>
bool FullyConnectedGradientOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, //  X
        float16, //  W
        float16, // dY
        float16, //  B
        float16, // dX
        float16, // dW
        float16, // dB
        float>(); // Math
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

#if CUDA_VERSION >= 9000

// Require these to be defined otherwise TensorCore FC ops will end
// up calling the default FC implementation which doesn't have
// fp16 support...
template <>
bool FullyConnectedOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, // X
        float16, // W
        float16, // B
        float16, // Y
        float>(); // Math
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

template <>
bool FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, //  X
        float16, //  W
        float16, // dY
        float16, //  B
        float16, // dX
        float16, // dW
        float16, // dB
        float>(); // Math
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

#endif

REGISTER_CUDA_OPERATOR(FC, FullyConnectedOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(FCGradient, FullyConnectedGradientOp<CUDAContext>);

#if CUDA_VERSION >= 9000
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FC,
    TENSORCORE,
    FullyConnectedOp<CUDAContext, TensorCoreEngine>);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FCGradient,
    TENSORCORE,
    FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>);
#endif

}  // namespace caffe2
