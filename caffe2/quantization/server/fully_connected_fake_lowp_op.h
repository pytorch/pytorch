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

#ifdef __x86_64__
#include <immintrin.h>
#endif
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// convert to float16 reducing mantissa, preserving exponent
void fp32_to_bfp16(const float* source, size_t size, float* dest);

// convert to float24 reducing mantissa, preserving exponent
void fp32_to_bfp24(const float* source, size_t size, float* dest);

// convert to float14 reducing mantissa, preserving exponent
void fp32_to_bfp14(const float* source, size_t size, float* dest);

void fp32_to_bfp16_scalar(const float* source, size_t size, float* dest);

// convert to IEEE float16
void fp32_to_fp16(const float* source, size_t size, float* dest);

// fp32 -> int32 -> += 1<< 15 -> fp32 -> truncation
void fp32_to_bfp16_round(const float* source, size_t size, float* dest);

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedFakeLowpFPOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedFakeLowpFPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            this->template GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedFakeLowpFPOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_B,
      typename T_Y,
      typename MATH>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;
  Tensor bias_multiplier_;

  bool float16_compute_;
};

template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedGradientFakeLowpFPOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedGradientFakeLowpFPOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            this->template GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedGradientFakeLowpFPOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_DY,
      typename T_B,
      typename T_DX,
      typename T_DW,
      typename T_DB,
      typename MATH>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  Tensor bias_multiplier_;
  bool float16_compute_;
};

} // namespace caffe2
